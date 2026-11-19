import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .brain import Brain, LLMBrain
from .cache import Cache, CachedBrain
from .config import Settings
from .graph import build_graph
from .metrics import RunMetrics
from .schemas import ResearchResponse
from .search import Searcher, WikipediaSearcher, make_http_client
from .state import RESET


@dataclass
class Agent:
    graph: object
    brain: Brain
    searcher: Searcher
    settings: Settings


@asynccontextmanager
async def _open_saver(path: str | None):
    if not path:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        yield saver


@asynccontextmanager
async def open_agent(settings: Settings, brain=None, searcher=None, checkpointer=None, persist: bool = True):
    """Builds the graph and everything it needs, and cleans it all up afterwards."""
    async with AsyncExitStack() as stack:
        cache = None
        if settings.cache_enabled:
            cache = Cache(settings.cache_path)
            stack.callback(cache.close)
        if searcher is None:
            client = await stack.enter_async_context(make_http_client())
            searcher = WikipediaSearcher(client, pages_per_query=settings.pages_per_query, cache=cache)
        if brain is None:
            brain = LLMBrain.from_settings(settings)
            if cache:
                brain = CachedBrain(brain, cache)
        if checkpointer is None and persist:
            checkpointer = await stack.enter_async_context(_open_saver(settings.checkpoint_path))
        graph = build_graph(
            brain,
            searcher,
            checkpointer=checkpointer,
            top_k=settings.top_k,
            llm_attempts=settings.node_retry_attempts,
        )
        yield Agent(graph, brain, searcher, settings)


def new_thread_id() -> str:
    return uuid.uuid4().hex


def initial_state(question: str, max_iterations: int, enable_critique: bool = True) -> dict:
    return {"question": question, "max_iterations": max_iterations, "enable_critique": enable_critique}


def run_config(thread_id: str, metrics: RunMetrics, recursion_limit: int = 40) -> dict:
    return {
        "recursion_limit": recursion_limit,
        "callbacks": [metrics],
        "configurable": {"thread_id": thread_id, "metrics": metrics},
    }


async def stream_run(graph, inputs: dict, config: dict):
    """Yields ("step", trace line) as nodes finish, then ("result", final state)."""
    final: dict = {}
    async for mode, chunk in graph.astream(inputs, config, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
            continue
        for update in chunk.values():
            for line in (update or {}).get("trace", []) if isinstance(update, dict) else []:
                if line != RESET:
                    yield "step", line
    yield "result", final


def to_response(state: dict, thread_id: str, metrics: RunMetrics) -> ResearchResponse:
    critique = state.get("critique")
    return ResearchResponse(
        thread_id=thread_id,
        answer=state.get("draft", ""),
        short_answer=state.get("short_answer", ""),
        sources=state.get("sources", []),
        iterations=state.get("iteration", 0),
        score=critique.score if critique else None,
        citations_valid=not state.get("citation_errors"),
        stop_reason=state["stop_reason"],
        trace=state.get("trace", []),
        metrics=metrics.summary(),
    )
