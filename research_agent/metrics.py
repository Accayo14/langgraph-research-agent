import time
from collections import Counter, defaultdict

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.config import get_config

from .schemas import Metrics


class RunMetrics(BaseCallbackHandler):
    """Collects LLM usage (via LangChain callbacks) and node timings for one run."""

    run_inline = True

    def __init__(self):
        self.started = time.perf_counter()
        self.llm_calls = 0
        self.failed_llm_calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.models: Counter[str] = Counter()
        self.node_ms: defaultdict[str, float] = defaultdict(float)

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None) or {}
                self.prompt_tokens += usage.get("input_tokens", 0)
                self.completion_tokens += usage.get("output_tokens", 0)
                if name := (msg.response_metadata or {}).get("model_name"):
                    self.models[name] += 1

    def on_llm_error(self, error, **kwargs):
        self.failed_llm_calls += 1

    def record_node(self, name: str, ms: float):
        self.node_ms[name] += ms

    def summary(self) -> Metrics:
        return Metrics(
            total_latency_ms=round((time.perf_counter() - self.started) * 1000, 1),
            node_latency_ms={k: round(v, 1) for k, v in self.node_ms.items()},
            llm_calls=self.llm_calls,
            failed_llm_calls=self.failed_llm_calls,
            cache_hits=self.cache_hits,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            models_used=dict(self.models),
        )


def current_metrics() -> RunMetrics | None:
    try:
        return get_config().get("configurable", {}).get("metrics")
    except RuntimeError:  # not inside a graph run
        return None
