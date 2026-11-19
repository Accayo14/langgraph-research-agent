import operator
from typing import Annotated, TypedDict

from .schemas import Critique, Source, Turn

# Put RESET first in an update to clear a list instead of appending to it (used at the start of each turn).
RESET = "__reset__"


def merge_sources(existing: list[Source], new: list[Source]) -> list[Source]:
    seen = {s.id for s in existing}
    merged = list(existing)
    for s in new:
        if s.id not in seen:
            seen.add(s.id)
            merged.append(s)
    return merged


def append_or_reset(existing: list[str], new: list[str]) -> list[str]:
    if new and new[0] == RESET:
        return list(new[1:])
    return existing + new


class ResearchState(TypedDict, total=False):
    question: str
    max_iterations: int
    enable_critique: bool
    history: Annotated[list[Turn], operator.add]
    pending_queries: list[str]
    turn_queries: Annotated[list[str], append_or_reset]
    searched_queries: Annotated[list[str], operator.add]
    evidence: Annotated[list[Source], merge_sources]
    sources: list[Source]
    draft: str
    short_answer: str
    citation_errors: list[str]
    critique: Critique | None
    approved: bool
    iteration: int
    stop_reason: str
    trace: Annotated[list[str], append_or_reset]
