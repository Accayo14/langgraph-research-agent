import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .brain import Brain, LLMBrain
from .config import Settings
from .graph import build_graph
from .schemas import ResearchRequest, ResearchResponse
from .search import Searcher, WikipediaSearcher, make_http_client


def _initial_state(req: ResearchRequest) -> dict:
    return {"question": req.question, "max_iterations": req.max_iterations}


def _run_config(settings: Settings) -> dict:
    return {"recursion_limit": settings.recursion_limit, "configurable": {"thread_id": str(uuid.uuid4())}}


def to_response(state: dict) -> ResearchResponse:
    critique = state.get("critique")
    return ResearchResponse(
        answer=state.get("draft", ""),
        sources=state.get("sources", []),
        iterations=state.get("iteration", 0),
        score=critique.score if critique else None,
        stop_reason=state["stop_reason"],
        trace=state.get("trace", []),
    )


def create_app(brain: Brain | None = None, searcher: Searcher | None = None, settings: Settings | None = None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = None
        s = searcher
        if s is None:
            client = make_http_client()
            s = WikipediaSearcher(client)
        app.state.graph = build_graph(brain or LLMBrain.from_settings(settings), s)
        yield
        if client:
            await client.aclose()

    app = FastAPI(title="LangGraph Research Agent", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": settings.model}

    @app.post("/research", response_model=ResearchResponse)
    async def research(req: ResearchRequest, request: Request):
        graph = request.app.state.graph
        try:
            async with asyncio.timeout(settings.request_timeout_s):
                final = await graph.ainvoke(_initial_state(req), _run_config(settings))
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Research timed out")
        return to_response(final)

    @app.post("/research/stream")
    async def research_stream(req: ResearchRequest, request: Request):
        # SSE stream
        graph = request.app.state.graph

        async def events():
            state: dict = {}
            async for chunk in graph.astream(_initial_state(req), _run_config(settings), stream_mode="values"):
                prev_len = len(state.get("trace", []))
                state = chunk
                for line in state.get("trace", [])[prev_len:]:
                    yield f"event: step\ndata: {json.dumps(line)}\n\n"
            yield f"event: result\ndata: {to_response(state).model_dump_json()}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def __getattr__(name: str):
    # so tests can import without api keys
    if name == "app":
        return create_app()
    raise AttributeError(name)
