from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from .brain import Brain
from .search import Searcher
from .state import ResearchState


def build_graph(brain: Brain, searcher: Searcher, checkpointer=None):
    async def plan(state: ResearchState) -> dict:
        p = await brain.plan(state["question"])
        return {"pending_queries": p.queries, "iteration": 0, "trace": [f"plan -> queries={p.queries}"]}

    async def search(state: ResearchState) -> dict:
        queries = state.get("pending_queries", [])
        found = await searcher.search_many(queries)
        return {
            "sources": found,
            "searched_queries": queries,
            "pending_queries": [],
            "trace": [f"search -> {len(found)} results for {queries}"],
        }

    async def draft(state: ResearchState) -> dict:
        critique = state.get("critique")
        text = await brain.write(
            state["question"],
            state.get("sources", []),
            previous_draft=state.get("draft"),
            feedback=critique.feedback if critique else None,
        )
        n = state.get("iteration", 0) + 1
        return {"draft": text, "iteration": n, "trace": [f"draft #{n}"]}

    async def critique(state: ResearchState) -> dict:
        c = await brain.critique(state["question"], state["draft"], state.get("sources", []))
        # dont search the same thing twice
        already = {q.lower() for q in state.get("searched_queries", [])}
        new_queries = [q for q in c.missing_info_queries if q.lower() not in already]
        return {
            "critique": c,
            "pending_queries": new_queries,
            "trace": [f"critique -> score={c.score} approved={c.approved} new_queries={new_queries}"],
        }

    def route_after_critique(state: ResearchState) -> Literal["search", "draft", "finalize"]:
        c = state["critique"]
        if c.approved or state["iteration"] >= state["max_iterations"]:
            return "finalize"
        if state.get("pending_queries"):
            return "search"
        return "draft"

    async def finalize(state: ResearchState) -> dict:
        reason = "approved" if state["critique"].approved else "max_iterations"
        return {"stop_reason": reason, "trace": [f"finalize -> {reason}"]}

    llm_retry = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0)

    g = StateGraph(ResearchState)
    g.add_node("plan", plan, retry_policy=llm_retry)
    g.add_node("search", search)
    g.add_node("draft", draft, retry_policy=llm_retry)
    g.add_node("critique", critique, retry_policy=llm_retry)
    g.add_node("finalize", finalize)

    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "draft")
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", route_after_critique, ["search", "draft", "finalize"])
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
