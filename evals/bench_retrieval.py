"""Retrieval latency benchmark (no LLM calls): sequential vs concurrent fetching, cold vs warm cache.

python -m evals.bench_retrieval
"""

import asyncio
import json
import statistics
import time
from pathlib import Path

from research_agent.cache import Cache
from research_agent.search import WikipediaSearcher, make_http_client

from .hotpotqa import DEFAULT_PATH

RESULTS = Path(__file__).parent / "results"


async def sequential(searcher, queries):
    return [await searcher.retrieve(q) for q in queries]


async def concurrent(searcher, queries):
    return await asyncio.gather(*(searcher.retrieve(q) for q in queries))


async def timed_run(fn, searcher, queries, passages: list) -> float:
    start = time.perf_counter()
    results = await fn(searcher, queries)
    passages.append(sum(map(len, results)))
    return time.perf_counter() - start


async def main(rounds: int = 3, n_queries: int = 8):
    items = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    queries = list(dict.fromkeys(t for i in items for t in i["supporting_titles"]))[:n_queries]
    seq, conc, cold, warm, passages = [], [], [], [], []
    failed = 0
    async with make_http_client() as client:
        for r in range(rounds):
            # alternate the order so neither mode always benefits from warmed-up connections
            order = [("seq", sequential), ("conc", concurrent)] if r % 2 == 0 else [("conc", concurrent), ("seq", sequential)]
            for name, fn in order:
                s = WikipediaSearcher(client)
                (seq if name == "seq" else conc).append(await timed_run(fn, s, queries, passages))
                failed += s.failed_requests
            cache = Cache()
            cached = WikipediaSearcher(client, cache=cache)
            cold.append(await timed_run(concurrent, cached, queries, passages))
            warm.append(await timed_run(concurrent, cached, queries, passages))
            failed += cached.failed_requests
            cache.close()

    med = statistics.median
    result = {
        "queries": queries,
        "rounds": rounds,
        "failed_requests": failed,
        "passages_per_run": sorted(set(passages)),
        "sequential_s": round(med(seq), 3),
        "concurrent_s": round(med(conc), 3),
        "concurrency_speedup": round(med(seq) / med(conc), 1),
        "cache_cold_s": round(med(cold), 3),
        "cache_warm_s": round(med(warm), 4),
        "cache_speedup": round(med(cold) / med(warm)),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_bench.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
