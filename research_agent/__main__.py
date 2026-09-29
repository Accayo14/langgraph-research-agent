import argparse
import asyncio
import logging

from .api import _initial_state, _run_config, to_response
from .brain import LLMBrain
from .config import Settings
from .graph import build_graph
from .schemas import ResearchRequest
from .search import WikipediaSearcher, make_http_client


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    req = ResearchRequest(question=args.question, max_iterations=args.max_iterations)
    settings = Settings()
    logging.basicConfig(format="    (%(message)s)")
    logging.getLogger("research_agent").setLevel(logging.INFO)
    brain = LLMBrain.from_settings(settings)
    print("model chain:", " -> ".join(brain.model_names))
    async with make_http_client() as client:
        graph = build_graph(brain, WikipediaSearcher(client))
        state: dict = {}
        async for state in graph.astream(_initial_state(req), _run_config(settings), stream_mode="values"):
            if state.get("trace"):
                print("  >", state["trace"][-1])

    res = to_response(state)
    print("\n" + res.answer + "\n")
    for i, s in enumerate(res.sources, 1):
        print(f"[{i}] {s.title} - {s.url}")
    print(f"\nstop_reason={res.stop_reason} iterations={res.iterations} score={res.score}")


if __name__ == "__main__":
    asyncio.run(main())
