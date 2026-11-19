import logging
from typing import Protocol, Sequence

from .schemas import Critique, Draft, Plan, Source, Turn

log = logging.getLogger(__name__)


class Brain(Protocol):
    async def plan(self, question: str, history: Sequence[Turn] = ()) -> Plan: ...

    async def write(
        self,
        question: str,
        sources: list[Source],
        previous_draft: str | None,
        feedback: str | None,
        history: Sequence[Turn] = (),
    ) -> Draft: ...

    async def critique(self, question: str, draft: str, sources: list[Source]) -> Critique: ...


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{i}] {s.title} ({s.url})\n{s.snippet}" for i, s in enumerate(sources, 1))


def format_history(history: Sequence[Turn]) -> str:
    if not history:
        return ""
    turns = "\n\n".join(f"Q: {t.question}\nA: {t.answer[:600]}" for t in history[-3:])
    return f"Conversation so far (use it to resolve references like 'he' or 'it'):\n{turns}\n\n"


PLAN_PROMPT = """{history}You are a research planner. Break the user's question into 1-4 short Wikipedia
search queries (article-title style, e.g. "CRISPR gene editing") that together gather the facts needed.
If the question involves several people, places or things, use one query per entity.

Question: {question}"""

WRITE_PROMPT = """{history}Answer the question using ONLY the numbered sources below. Cite sources inline
using plain square-bracket numbers like [1] or [2] (no other citation format). Ignore sources that are
irrelevant to the question. If the sources do not contain the answer, say so rather than guessing.
Keep the answer concise (80-250 words). Also give short_answer: the direct answer in as few words as
possible (e.g. "yes", a name, a date), without citations.

Question: {question}

Sources:
{sources}
{revision}"""

REVISION_BLOCK = """
Your previous draft:
{draft}

Fix these problems in your rewrite:
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
        llms = list(zip([init_chat_model(m, temperature=0, **kwargs) for m in models], models))

        self._planner = self._chain([self._structured(m, name, Plan) for m, name in llms])
        self._writer = self._chain([self._structured(m, name, Draft) for m, name in llms])
        self._critic = self._chain([self._structured(m, name, Critique) for m, name in llms])

    @staticmethod
    def _chain(runnables):
        primary, *rest = runnables
        return primary.with_fallbacks(rest) if rest else primary

    @staticmethod
    def _structured(llm, name: str, schema):
        from langchain_core.runnables import RunnableLambda

        def check(out: dict):
            parsed = out.get("parsed")
            if parsed is None:
                # raise so with_fallbacks moves to next model
                raise ValueError(f"{name} returned invalid {schema.__name__}: {out.get('parsing_error')}")
            if isinstance(parsed, Draft) and not parsed.answer.strip():
                raise ValueError(f"{name} returned an empty answer")
            log.info("%s produced by %s", schema.__name__, name)
            return parsed

        return llm.with_structured_output(schema, include_raw=True) | RunnableLambda(check)

    @classmethod
    def from_settings(cls, settings) -> "LLMBrain":
        return cls(
            settings.model_chain(),
            base_url=settings.base_url,
            api_key=settings.openai_api_key,
            max_retries=settings.llm_max_retries,
        )

    async def plan(self, question, history=()) -> Plan:
        return await self._planner.ainvoke(
            PLAN_PROMPT.format(question=question, history=format_history(history))
        )

    async def write(self, question, sources, previous_draft, feedback, history=()) -> Draft:
        revision = (
            REVISION_BLOCK.format(draft=previous_draft, feedback=feedback) if previous_draft and feedback else ""
        )
        return await self._writer.ainvoke(
            WRITE_PROMPT.format(
                question=question,
                sources=format_sources(sources),
                revision=revision,
                history=format_history(history),
            )
        )

    async def critique(self, question, draft, sources) -> Critique:
        return await self._critic.ainvoke(
            CRITIQUE_PROMPT.format(question=question, sources=format_sources(sources), draft=draft)
        )
