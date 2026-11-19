import re
import time
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from .brain import Brain
from .metrics import current_metrics
from .retrieval import rank_passages
from .schemas import Turn
from .search import Searcher
from .state import RESET, ResearchState

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def check_citations(text: str, n_sources: int) -> list[str]:
    if n_sources == 0:
        return []
    cited = [int(n) for group in _CITATION.findall(text) for n in group.split(",")]
    if not cited:
        return ["The answer has no citations. Cite the sources you used as [1], [2], ..."]
    bad = sorted({n for n in cited if not 1 <= n <= n_sources})
    return [f"Citation [{n}] does not exist; only sources [1]-[{n_sources}] are available." for n in bad]


def timed(name, fn):
    async def node(state):
        start = time.perf_counter()
        try:
            return await fn(state)
        finally:
            if m := current_metrics():
                m.record_node(name, (time.perf_counter() - start) * 1000)

    return node


def build_graph(brain: Brain, searcher: Searcher, checkpointer=None, top_k: int = 8, llm_attempts: int = 3):
    def fan_out(state: ResearchState) -> list[Send]:
        # one retrieve task per query, run in parallel
        return [Send("retrieve", {"query": q}) for q in state["pending_queries"]]

    async def plan(state: ResearchState) -> dict:
        p = await brain.plan(state["question"], state.get("history", []))
        return {
            "pending_queries": p.queries,
            "turn_queries": [RESET, *p.queries],
            "iteration": 0,
            "draft": "",
            "short_answer": "",
            "critique": None,
            "approved": False,
            "citation_errors": [],
            "sources": [],
            "stop_reason": "",
            "trace": [RESET, f"plan -> queries={p.queries}"],
        }

    async def retrieve(task: dict) -> dict:
        found = await searcher.retrieve(task["query"])
        return {
            "evidence": found,
            "searched_queries": [task["query"]],
            "trace": [f"retrieve '{task['query']}' -> {len(found)} passages"],
        }

    async def select(state: ResearchState) -> dict:
        query = " ".join([state["question"], *state.get("turn_queries", [])])
        chosen = rank_passages(query, state.get("evidence", []), k=top_k)
        titles = sorted({s.title for s in chosen})
        return {"sources": chosen, "pending_queries": [], "trace": [f"select -> {len(chosen)} passages from {titles}"]}

    async def draft(state: ResearchState) -> dict:
        critique = state.get("critique")
        problems = state.get("citation_errors") or ([critique.feedback] if critique else [])
        d = await brain.write(
            state["question"],
            state.get("sources", []),
            previous_draft=state.get("draft") or None,
            feedback="\n".join(problems) or None,
            history=state.get("history", []),
        )
        n = state.get("iteration", 0) + 1
        return {"draft": d.answer, "short_answer": d.short_answer, "iteration": n, "approved": False, "trace": [f"draft #{n}"]}

    async def verify(state: ResearchState) -> dict:
        errors = check_citations(state["draft"], len(state.get("sources", [])))
        return {"citation_errors": errors, "trace": [f"verify -> {'ok' if not errors else errors}"]}

    def route_after_verify(state: ResearchState) -> Literal["draft", "critique", "finalize"]:
        can_revise = state["iteration"] < state["max_iterations"]
        if state["citation_errors"] and can_revise:
            return "draft"
        # the critic only runs if its verdict can still change something
        if not state.get("enable_critique", True) or not can_revise:
            return "finalize"
        return "critique"

    async def critique(state: ResearchState) -> dict:
        c = await brain.critique(state["question"], state["draft"], state.get("sources", []))
        already = {q.lower() for q in state.get("searched_queries", [])}
        new_queries = [q for q in c.missing_info_queries if q.lower() not in already]
        return {
            "critique": c,
            "approved": c.approved,
            "pending_queries": new_queries,
            "turn_queries": new_queries,
            "trace": [f"critique -> score={c.score} approved={c.approved} new_queries={new_queries}"],
        }

    def route_after_critique(state: ResearchState):
        if state["approved"] or state["iteration"] >= state["max_iterations"]:
            return "finalize"
        if state.get("pending_queries"):
            return fan_out(state)
        return "draft"

    async def finalize(state: ResearchState) -> dict:
        if state.get("approved"):
            reason = "approved"
        elif not state.get("enable_critique", True):
            reason = "critique_disabled"
        else:
            reason = "max_iterations"
        return {
            "stop_reason": reason,
            "history": [Turn(question=state["question"], answer=state["draft"])],
            "trace": [f"finalize -> {reason}"],
        }

    llm_retry = RetryPolicy(max_attempts=llm_attempts, initial_interval=2.0, backoff_factor=2.0)

    g = StateGraph(ResearchState)
    g.add_node("plan", timed("plan", plan), retry_policy=llm_retry)
    g.add_node("retrieve", timed("retrieve", retrieve))
    g.add_node("select", timed("select", select))
    g.add_node("draft", timed("draft", draft), retry_policy=llm_retry)
    g.add_node("verify", timed("verify", verify))
    g.add_node("critique", timed("critique", critique), retry_policy=llm_retry)
    g.add_node("finalize", timed("finalize", finalize))

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["retrieve"])
    g.add_edge("retrieve", "select")
    g.add_edge("select", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["draft", "critique", "finalize"])
    g.add_conditional_edges("critique", route_after_critique, ["retrieve", "draft", "finalize"])
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
