import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import Settings
from .metrics import RunMetrics
from .runner import initial_state, new_thread_id, open_agent, run_config, stream_run, to_response
from .schemas import ResearchRequest, ResearchResponse, ThreadHistory


def create_app(brain=None, searcher=None, settings: Settings | None = None, checkpointer=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with open_agent(settings, brain=brain, searcher=searcher, checkpointer=checkpointer) as agent:
            app.state.agent = agent
            yield

    app = FastAPI(title="LangGraph Research Agent", version="0.2.0", lifespan=lifespan)

    def _start(req: ResearchRequest):
        thread_id = req.thread_id or new_thread_id()
        metrics = RunMetrics()
        return thread_id, metrics, run_config(thread_id, metrics, settings.recursion_limit)

    @app.get("/health")
    async def health():
        return {"status": "ok", "models": settings.model_chain()}

    @app.post("/research", response_model=ResearchResponse)
    async def research(req: ResearchRequest, request: Request):
        graph = request.app.state.agent.graph
        thread_id, metrics, config = _start(req)
        try:
            async with asyncio.timeout(settings.request_timeout_s):
                final = await graph.ainvoke(initial_state(req.question, req.max_iterations), config)
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Research timed out")
        return to_response(final, thread_id, metrics)

    @app.post("/research/stream")
    async def research_stream(req: ResearchRequest, request: Request):
        graph = request.app.state.agent.graph
        thread_id, metrics, config = _start(req)

        async def events():
            async for kind, payload in stream_run(graph, initial_state(req.question, req.max_iterations), config):
                if kind == "step":
                    yield f"event: step\ndata: {json.dumps(payload)}\n\n"
                else:
                    yield f"event: result\ndata: {to_response(payload, thread_id, metrics).model_dump_json()}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/threads/{thread_id}", response_model=ThreadHistory)
    async def thread_history(thread_id: str, request: Request):
        snapshot = await request.app.state.agent.graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Unknown thread")
        return ThreadHistory(thread_id=thread_id, turns=snapshot.values.get("history", []))

    return app


def __getattr__(name: str):
    # so tests can import without api keys
    if name == "app":
        return create_app()
    raise AttributeError(name)
