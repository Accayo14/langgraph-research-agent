import operator
from typing import Annotated, TypedDict

from .schemas import Critique, Source


def merge_sources(existing: list[Source], new: list[Source]) -> list[Source]:
    seen = {s.url for s in existing}
    merged = list(existing)
    for s in new:
        if s.url not in seen:
            seen.add(s.url)
            merged.append(s)
    return merged


class ResearchState(TypedDict, total=False):
    question: str
    max_iterations: int
    pending_queries: list[str]
    searched_queries: Annotated[list[str], operator.add]
    sources: Annotated[list[Source], merge_sources]
    draft: str
    critique: Critique | None
    iteration: int
    stop_reason: str
    trace: Annotated[list[str], operator.add]
