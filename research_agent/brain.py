import logging
from typing import Protocol

from .schemas import Critique, Plan, Source

log = logging.getLogger(__name__)


class Brain(Protocol):
    async def plan(self, question: str) -> Plan: ...

    async def write(
        self, question: str, sources: list[Source], previous_draft: str | None, feedback: str | None
    ) -> str: ...

    async def critique(self, question: str, draft: str, sources: list[Source]) -> Critique: ...


def _format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{i}] {s.title} ({s.url})\n{s.snippet}" for i, s in enumerate(sources, 1))


PLAN_PROMPT = """You are a research planner. Break the user's question into 1-4 short Wikipedia
search queries (article-title style, e.g. "CRISPR gene editing") that together gather the facts needed.

Question: {question}"""

WRITE_PROMPT = """Answer the question using ONLY the numbered sources below. Cite sources inline
using plain square-bracket numbers like [1] or [2] (no other citation format). Ignore sources that are
irrelevant to the question. If the sources do not contain something, say so rather than guessing. Be concise
(150-300 words).

Question: {question}

Sources:
{sources}
{revision}"""

REVISION_BLOCK = """
Your previous draft:
{draft}

A reviewer gave this feedback. Rewrite the answer to address it:
{feedback}"""

CRITIQUE_PROMPT = """You are a strict reviewer. Judge the draft answer against the sources.
Check: factual accuracy vs. sources, completeness w.r.t. the question, and that claims are cited.
Approve only if it is genuinely good (score >= 8). If facts are missing from the sources entirely,
propose up to 3 new search queries in missing_info_queries; otherwise leave it empty.

Question: {question}

Sources:
{sources}

Draft:
{draft}"""


class LLMBrain:
    def __init__(self, models: list[str], **kwargs):
        from langchain.chat_models import init_chat_model

        if not models:
            raise ValueError("at least one model is required")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model_names = models
        llms = [init_chat_model(m, temperature=0, **kwargs) for m in models]

        self._llm = self._chain([self._tagged(m, name) for m, name in zip(llms, models)])
        self._planner = self._chain([self._structured(m, name, Plan) for m, name in zip(llms, models)])
        self._critic = self._chain([self._structured(m, name, Critique) for m, name in zip(llms, models)])

    @staticmethod
    def _chain(runnables):
        primary, *rest = runnables
        return primary.with_fallbacks(rest) if rest else primary

    @staticmethod
    def _tagged(llm, name: str):
        from langchain_core.runnables import RunnableLambda

        def check(msg):
            text = msg.text if isinstance(getattr(msg, "text", None), str) else str(msg.content)
            if not text.strip():
                raise ValueError(f"{name} returned an empty response")
            log.info("answered by %s", name)
            return text

        return llm | RunnableLambda(check)

    @staticmethod
    def _structured(llm, name: str, schema):
        from langchain_core.runnables import RunnableLambda

        def check(out: dict):
            if out.get("parsed") is None:
                # raise so with_fallbacks moves to next model
                raise ValueError(f"{name} returned invalid {schema.__name__}: {out.get('parsing_error')}")
            log.info("%s produced by %s", schema.__name__, name)
            return out["parsed"]

        return llm.with_structured_output(schema, include_raw=True) | RunnableLambda(check)

    @classmethod
    def from_settings(cls, settings) -> "LLMBrain":
        return cls(
            settings.model_chain(),
            base_url=settings.base_url,
            api_key=settings.openai_api_key,
            max_retries=settings.llm_max_retries,
        )

    async def plan(self, question: str) -> Plan:
        return await self._planner.ainvoke(PLAN_PROMPT.format(question=question))

    async def write(self, question, sources, previous_draft, feedback) -> str:
        revision = (
            REVISION_BLOCK.format(draft=previous_draft, feedback=feedback) if previous_draft and feedback else ""
        )
        return await self._llm.ainvoke(
            WRITE_PROMPT.format(question=question, sources=_format_sources(sources), revision=revision)
        )

    async def critique(self, question, draft, sources) -> Critique:
        return await self._critic.ainvoke(
            CRITIQUE_PROMPT.format(question=question, sources=_format_sources(sources), draft=draft)
        )