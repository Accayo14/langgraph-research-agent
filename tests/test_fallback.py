import asyncio

import httpx
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from research_agent import openrouter
from research_agent.brain import LLMBrain
from research_agent.config import Settings
from research_agent.schemas import Plan


def _rate_limited(_):
    raise RuntimeError("429 rate limited")


def test_text_call_falls_through_failed_and_empty_models():
    chain = LLMBrain._chain(
        [
            LLMBrain._tagged(RunnableLambda(_rate_limited), "a"),
            LLMBrain._tagged(RunnableLambda(lambda _: AIMessage(content="  ")), "b"),
            LLMBrain._tagged(RunnableLambda(lambda _: AIMessage(content="hello")), "c"),
        ]
    )
    assert asyncio.run(chain.ainvoke("hi")) == "hello"


def test_structured_call_falls_back_on_invalid_output():
    class FakeStructured:
        def __init__(self, out):
            self.out = out

        def with_structured_output(self, schema, include_raw):
            return RunnableLambda(lambda _: self.out)

    bad = FakeStructured({"parsed": None, "parsing_error": "not json"})
    good = FakeStructured({"parsed": Plan(queries=["x"])})
    chain = LLMBrain._chain([LLMBrain._structured(m, n, Plan) for m, n in [(bad, "bad"), (good, "good")]])
    assert asyncio.run(chain.ainvoke("q")).queries == ["x"]


def test_model_chain_dedupes_and_caps(monkeypatch):
    monkeypatch.setattr(openrouter, "discover_free_models", lambda: ["m/a:free", "m/new:free", "m/new2:free"])
    s = Settings(
        _env_file=None,
        model="openai:m/a:free",
        fallback_models=" openai:m/b:free , openai:m/a:free ",
        auto_discover_free_models=True,
        max_fallback_models=2,
    )
    assert s.model_chain() == ["openai:m/a:free", "openai:m/b:free", "openai:m/new:free"]


def test_discovery_keeps_only_free_models_with_structured_output(monkeypatch):
    data = {
        "data": [
            {"id": "a:free", "context_length": 10, "supported_parameters": ["tools", "structured_outputs"]},
            {"id": "b:free", "context_length": 99, "supported_parameters": ["tools", "structured_outputs"]},
            {"id": "c:free", "context_length": 50, "supported_parameters": ["tools"]},
            {"id": "paid", "context_length": 50, "supported_parameters": ["tools", "structured_outputs"]},
        ]
    }
    req = httpx.Request("GET", openrouter.MODELS_URL)
    monkeypatch.setattr(openrouter.httpx, "get", lambda *a, **k: httpx.Response(200, json=data, request=req))
    assert openrouter.discover_free_models() == ["b:free", "a:free"]
