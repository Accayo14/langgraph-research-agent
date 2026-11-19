import argparse
import asyncio
import logging

from .config import Settings
from .metrics import RunMetrics
from .runner import initial_state, new_thread_id, open_agent, run_config, stream_run, to_response
from .schemas import ResearchRequest


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the research agent a question.")
    parser.add_argument("question")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--thread", help="thread id from a previous run, to ask a follow-up")
    args = parser.parse_args()

    req = ResearchRequest(question=args.question, max_iterations=args.max_iterations, thread_id=args.thread)
    settings = Settings()
    logging.basicConfig(format="    (%(message)s)")
    logging.getLogger("research_agent").setLevel(logging.INFO)

    thread_id = req.thread_id or new_thread_id()
    metrics = RunMetrics()
    async with open_agent(settings) as agent:
        print("model chain:", " -> ".join(agent.brain.model_names))
        state: dict = {}
        config = run_config(thread_id, metrics, settings.recursion_limit)
        async for kind, payload in stream_run(agent.graph, initial_state(req.question, req.max_iterations), config):
            if kind == "step":
                print("  >", payload)
            else:
                state = payload

    res = to_response(state, thread_id, metrics)
    print(f"\n{res.answer}\n\nshort answer: {res.short_answer}\n")
    for i, s in enumerate(res.sources, 1):
        print(f"[{i}] {s.title} - {s.url}")
    m = res.metrics
    print(
        f"\nstop_reason={res.stop_reason} iterations={res.iterations} score={res.score} "
        f"citations_valid={res.citations_valid}\n"
        f"latency={m.total_latency_ms / 1000:.1f}s llm_calls={m.llm_calls} failed={m.failed_llm_calls} "
        f"cache_hits={m.cache_hits} tokens={m.prompt_tokens}+{m.completion_tokens} models={m.models_used}"
    )
    print(f"\nthread: {thread_id}  (ask a follow-up with --thread {thread_id})")


if __name__ == "__main__":
    asyncio.run(main())
