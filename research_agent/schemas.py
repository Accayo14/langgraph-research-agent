from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    id: str
    title: str
    url: str
    snippet: str


class Turn(BaseModel):
    question: str
    answer: str


class Plan(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=4,
        description="1-4 short, distinct Wikipedia search queries that together cover the question.",
    )


class Draft(BaseModel):
    answer: str = Field(description="The full answer, citing sources inline as [1], [2].")
    short_answer: str = Field(
        description="The direct answer in as few words as possible, e.g. 'yes', 'no', a name, a number "
        "or a date. No citations."
    )


class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality of the draft, 1-10.")
    approved: bool = Field(description="True only if the draft is accurate, complete and well cited.")
    feedback: str = Field(description="Concrete, actionable instructions for improving the draft.")
    missing_info_queries: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="New search queries to run ONLY if the sources lack needed facts; empty otherwise.",
    )


class ResearchRequest(BaseModel):
    question: str = Field(min_length=10, max_length=500, examples=["How does CRISPR-Cas9 edit genes?"])
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max number of drafts.")
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Pass the thread_id from a previous response to ask a follow-up question.",
    )

    @field_validator("question")
    @classmethod
    def must_be_meaningful(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 10:
            raise ValueError("question must contain at least 10 non-whitespace characters")
        return v


class Metrics(BaseModel):
    total_latency_ms: float
    node_latency_ms: dict[str, float]
    llm_calls: int
    failed_llm_calls: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    models_used: dict[str, int]


class ResearchResponse(BaseModel):
    thread_id: str
    answer: str
    short_answer: str
    sources: list[Source]
    iterations: int
    score: int | None
    citations_valid: bool
    stop_reason: Literal["approved", "max_iterations", "critique_disabled"]
    trace: list[str]
    metrics: Metrics


class ThreadHistory(BaseModel):
    thread_id: str
    turns: list[Turn]
