from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from research_agent.graph import check_citations
from research_agent.retrieval import rank_passages
from research_agent.schemas import Source


class BaselineState(TypedDict, total=False):
    question: str
    sources: list[Source]
    draft: str
    short_answer: str
    citation_errors: list[str]


def build_baseline(brain, searcher, top_k: int = 8):
    """Single-pass RAG: search Wikipedia with the raw question, rank passages, answer once.

    Same retriever, ranker, prompt and models as the agent, just no planning, loops or critic.
    """

    async def answer(state: BaselineState) -> dict:
        q = state["question"]
        chosen = rank_passages(q, await searcher.retrieve(q), k=top_k)
        d = await brain.write(q, chosen, None, None)
        return {
            "sources": chosen,
            "draft": d.answer,
            "short_answer": d.short_answer,
            "citation_errors": check_citations(d.answer, len(chosen)),
        }

    g = StateGraph(BaselineState)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile()
