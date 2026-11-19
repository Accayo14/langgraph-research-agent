"""Evaluate the agent against a single-pass RAG baseline on the HotpotQA sample.

python -m evals.run --variants baseline agent --limit 6

Results are appended to evals/results/<name>.jsonl, so an interrupted run (e.g. after hitting a
daily rate limit) picks up where it left off. A summary table is written to <name>.md.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from research_agent.config import Settings
from research_agent.metrics import RunMetrics
from research_agent.runner import initial_state, open_agent, run_config

from .baseline import build_baseline
from .hotpotqa import DEFAULT_PATH
from .scoring import answer_found, exact_match, f1, support_recall

RESULTS = Path(__file__).parent / "results"
VARIANTS = ("baseline", "agent", "agent_no_critique")


async def run_one(agent, baseline, variant: str, item: dict, max_iterations: int) -> dict:
    metrics = RunMetrics()
    config = run_config(f"eval-{variant}-{item['id']}", metrics)
    record = {"id": item["id"], "variant": variant, "type": item["type"], "question": item["question"], "gold": item["answer"]}
    try:
        if variant == "baseline":
            state = await baseline.ainvoke({"question": item["question"]}, config)
        else:
            inputs = initial_state(item["question"], max_iterations, enable_critique=variant == "agent")
            state = await agent.graph.ainvoke(inputs, config)
    except Exception as e:  # keep going; the failure is recorded
        m = metrics.summary()
        return record | {"error": f"{type(e).__name__}: {e}"[:300], "llm_calls": m.llm_calls, "failed_llm_calls": m.failed_llm_calls}

    m = metrics.summary()
    short, answer = state.get("short_answer", ""), state.get("draft", "")
    sources = state.get("sources", [])
    critique = state.get("critique")
    return record | {
        "short_answer": short,
        "answer": answer,
        "em": exact_match(short, item["answer"]),
        "f1": f1(short, item["answer"]),
        "answer_in_text": answer_found(item["answer"], short, answer, item["question"]),
        "support_recall": support_recall(item["supporting_titles"], [s.title for s in sources]),
        "source_titles": sorted({s.title for s in sources}),
        "citations_valid": not state.get("citation_errors"),
        "iterations": state.get("iteration", 1),
        "approved": bool(state.get("approved")),
        "first_score": critique.score if critique else None,
        "stop_reason": state.get("stop_reason", "single_pass"),
        "latency_s": round(m.total_latency_ms / 1000, 2),
        "llm_calls": m.llm_calls,
        "failed_llm_calls": m.failed_llm_calls,
        "cache_hits": m.cache_hits,
        "tokens": m.prompt_tokens + m.completion_tokens,
        "models": m.models_used,
        "trace": state.get("trace", []),
    }


def rescore(r: dict) -> dict:
    """Recompute answer metrics from a saved record, without re-running the model."""
    if "error" in r:
        return r
    short, answer, gold = r["short_answer"], r["answer"], r["gold"]
    return r | {
        "em": exact_match(short, gold),
        "f1": f1(short, gold),
        "answer_in_text": answer_found(gold, short, answer, r["question"]),
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        rs = [r for r in records if r["variant"] == variant]
        if not rs:
            continue
        ok = [r for r in rs if "error" not in r]
        mean = lambda k: statistics.mean(r[k] for r in ok) if ok else float("nan")  # noqa: E731
        rows.append(
            {
                "variant": variant,
                "n": len(rs),
                "errors": len(rs) - len(ok),
                "exact_match": mean("em"),
                "f1": mean("f1"),
                "answer_in_text": mean("answer_in_text"),
                "support_recall": mean("support_recall"),
                "citations_valid": mean("citations_valid"),
                "avg_iterations": mean("iterations"),
                "avg_llm_calls": mean("llm_calls"),
                "avg_tokens": mean("tokens"),
                "median_latency_s": statistics.median(r["latency_s"] for r in ok) if ok else float("nan"),
                "failed_calls_recovered": sum(r["failed_llm_calls"] for r in ok),
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    lines = [
        "| Variant | n | EM | F1 | Answer found | Gold-page recall | Valid citations | Avg drafts | Avg LLM calls | Avg tokens | Median latency | Failed calls recovered | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['n']} | {pct(r['exact_match'])} | {r['f1']:.2f} | {pct(r['answer_in_text'])} | "
            f"{pct(r['support_recall'])} | {pct(r['citations_valid'])} | {r['avg_iterations']:.1f} | "
            f"{r['avg_llm_calls']:.1f} | {r['avg_tokens']:.0f} | {r['median_latency_s']:.1f}s | "
            f"{r['failed_calls_recovered']} | {r['errors']} |"
        )
    return "\n".join(lines)


async def run_all(settings, args, items, done, save) -> list[dict]:
    finished = {(r["id"], r["variant"]) for r in done if "error" not in r}
    spent = 0
    async with open_agent(settings, persist=False) as agent:
        print("model chain:", " -> ".join(agent.brain.model_names))
        baseline = build_baseline(agent.brain, agent.searcher, top_k=settings.top_k)
        for item in items:
            for variant in args.variants:
                if (item["id"], variant) in finished:
                    continue
                if spent >= args.max_llm_calls:
                    print(f"stopping: LLM call budget of {args.max_llm_calls} reached")
                    return done
                started = time.perf_counter()
                rec = await run_one(agent, baseline, variant, item, args.max_iterations)
                spent += rec.get("llm_calls", 0) + rec.get("failed_llm_calls", 0)
                done = [r for r in done if (r["id"], r["variant"]) != (item["id"], variant)] + [rec]
                save(done)
                status = rec.get("error") or f"pred={rec['short_answer']!r} gold={item['answer']!r} f1={rec['f1']:.2f}"
                print(f"[{variant:>17}] {item['question'][:60]:<60} {status} ({time.perf_counter() - started:.1f}s, {spent} calls so far)")
    return done


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["baseline", "agent"], choices=VARIANTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-llm-calls", type=int, default=40, help="stop before spending more than this")
    parser.add_argument("--name", default="hotpotqa")
    parser.add_argument("--model", help="override MODEL for this run")
    parser.add_argument("--fallback-models", help="override FALLBACK_MODELS for this run")
    parser.add_argument("--rescore-only", action="store_true", help="recompute metrics from saved results, no LLM calls")
    args = parser.parse_args()

    overrides = {"auto_discover_free_models": False, "llm_max_retries": 0, "node_retry_attempts": 2}
    if args.model:
        overrides["model"] = args.model
    if args.fallback_models is not None:
        overrides["fallback_models"] = args.fallback_models
    settings = Settings(**overrides)

    items = json.loads(args.data.read_text(encoding="utf-8"))[: args.limit]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.name}.jsonl"
    done = []
    if out.exists():
        done = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    save = lambda records: out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")  # noqa: E731

    if args.rescore_only:
        done = [rescore(r) for r in done]
        save(done)
    else:
        done = await run_all(settings, args, items, done, save)

    rows = summarize([r for r in done if any(r["id"] == i["id"] for i in items)])
    table = to_markdown(rows)
    (RESULTS / f"{args.name}.md").write_text(table + "\n", encoding="utf-8")
    (RESULTS / f"{args.name}_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    asyncio.run(main())
