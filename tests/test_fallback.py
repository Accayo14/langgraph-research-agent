import asyncio

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from research_agent import openrouter
from research_agent.brain import LLMBrain
from research_agent.config import Settings
from research_agent.schemas import Draft, Plan


class FakeStructured:
    def __init__(self, out):
        self.out = out

    def with_structured_output(self, schema, include_raw):
        if isinstance(self.out, Exception):
            def fail(_):
                raise self.out

            return RunnableLambda(fail)
        return RunnableLambda(lambda _: self.out)


def chain_of(schema, *models):
    return LLMBrain._chain([LLMBrain._structured(m, f"m{i}", schema) for i, m in enumerate(models)])


def test_falls_back_on_error_invalid_output_and_empty_answer():
    chain = chain_of(
        Draft,
        FakeStructured(RuntimeError("429 rate limited")),
        FakeStructured({"parsed": None, "parsing_error": "not json"}),
        FakeStructured({"parsed": Draft(answer="  ", short_answer="x")}),
        FakeStructured({"parsed": Draft(answer="hello [1]", short_answer="hello")}),
    )
    assert asyncio.run(chain.ainvoke("q")).short_answer == "hello"


def test_single_model_error_propagates():
    chain = chain_of(Plan, FakeStructured(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        asyncio.run(chain.ainvoke("q"))


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
