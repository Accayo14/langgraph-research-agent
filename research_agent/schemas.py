from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class Plan(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=4,
        description="1-4 short, distinct Wikipedia search queries that together cover the question.",
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
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max draft/critique cycles.")

    @field_validator("question")
    @classmethod
    def must_be_meaningful(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 10:
            raise ValueError("question must contain at least 10 non-whitespace characters")
        return v


class ResearchResponse(BaseModel):
    answer: str
    sources: list[Source]
    iterations: int
    score: int | None
    stop_reason: Literal["approved", "max_iterations"]
    trace: list[str]
