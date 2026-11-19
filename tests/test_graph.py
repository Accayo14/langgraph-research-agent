import asyncio
import time

from conftest import FakeBrain, FakeSearcher, bad, ok
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.graph import build_graph, check_citations
from research_agent.metrics import RunMetrics
from research_agent.runner import initial_state, run_config, stream_run
from research_agent.schemas import Draft


def run(brain, searcher, max_iterations=3, enable_critique=True, checkpointer=None, thread="t1", question="What is X and why?"):
    graph = build_graph(brain, searcher, checkpointer=checkpointer)
    metrics = RunMetrics()
    state = asyncio.run(graph.ainvoke(initial_state(question, max_iterations, enable_critique), run_config(thread, metrics)))
    return state, metrics


def test_approved_first_time_takes_straight_path():
    brain, s = FakeBrain([ok()]), FakeSearcher()
    out, _ = run(brain, s)
    assert out["stop_reason"] == "approved"
    assert out["iteration"] == 1
    assert sorted(s.calls) == ["topic A", "topic B"]
    assert brain.calls == {"plan": 1, "write": 1, "critique": 1}


def test_revision_loop_goes_back_to_draft_with_feedback():
    brain = FakeBrain([bad(), ok()])
    out, _ = run(brain, FakeSearcher())
    assert out["iteration"] == 2
    assert out["short_answer"] == "v2"
    assert brain.feedback == [None, "improve"]


def test_research_loop_fans_out_again_and_merges_evidence():
    s = FakeSearcher()
    out, _ = run(FakeBrain([bad(["topic C"]), ok()]), s)
    assert s.calls[2:] == ["topic C"]
    assert {e.title for e in out["evidence"]} == {"topic A", "topic B", "topic C"}


def test_retrieval_runs_in_parallel():
    brain = FakeBrain([ok()], queries=["a", "b", "c", "d"])
    start = time.perf_counter()
    run(brain, FakeSearcher(delay=0.3))
    assert time.perf_counter() - start < 0.9  # sequential would be >= 1.2s


def test_repeated_queries_are_not_searched_again():
    s = FakeSearcher()
    run(FakeBrain([bad(["Topic A"]), ok()]), s)
    assert len(s.calls) == 2


def test_loop_stops_at_max_iterations_without_a_useless_final_critique():
    brain = FakeBrain([bad()] * 10)
    out, _ = run(brain, FakeSearcher(), max_iterations=3)
    assert out["stop_reason"] == "max_iterations"
    assert brain.calls["write"] == 3
    assert brain.calls["critique"] == 2


def test_bad_citations_are_fixed_before_the_critic_is_asked():
    drafts = [Draft(answer="no citations here", short_answer="x"), Draft(answer="fixed [1]", short_answer="x")]
    brain = FakeBrain([ok()], drafts=drafts)
    out, _ = run(brain, FakeSearcher())
    assert out["draft"] == "fixed [1]"
    assert brain.calls == {"plan": 1, "write": 2, "critique": 1}
    assert "no citations" in brain.feedback[1]


def test_critique_can_be_disabled():
    brain = FakeBrain([])
    out, _ = run(brain, FakeSearcher(), enable_critique=False)
    assert out["stop_reason"] == "critique_disabled"
    assert brain.calls["critique"] == 0


def test_follow_up_questions_use_thread_memory():
    saver, brain = InMemorySaver(), FakeBrain([ok(), ok()])
    run(brain, FakeSearcher(), checkpointer=saver, question="Who founded SpaceX?")
    out, _ = run(brain, FakeSearcher(), checkpointer=saver, question="When was he born?")
    assert [t.question for t in brain.histories[1]] == ["Who founded SpaceX?"]
    assert len(out["history"]) == 2
    assert out["trace"][0].startswith("plan")  # trace restarts each turn
    assert out["iteration"] == 1


def test_metrics_record_node_timings():
    _, metrics = run(FakeBrain([ok()]), FakeSearcher())
    assert {"plan", "retrieve", "select", "draft", "verify", "critique", "finalize"} <= metrics.node_ms.keys()


def test_stream_run_yields_steps_then_result():
    graph = build_graph(FakeBrain([ok()]), FakeSearcher())

    async def collect():
        return [e async for e in stream_run(graph, initial_state("What is X and why?", 3), run_config("t", RunMetrics()))]

    events = asyncio.run(collect())
    steps = [p for k, p in events if k == "step"]
    assert steps[0].startswith("plan") and steps[-1].startswith("finalize")
    assert events[-1][0] == "result" and events[-1][1]["stop_reason"] == "approved"


def test_check_citations():
    assert check_citations("A [1] and B [2, 3].", 3) == []
    assert check_citations("No cites.", 3)[0].startswith("The answer has no citations")
    assert check_citations("Bad [4] [0] [2]", 3) == [
        "Citation [0] does not exist; only sources [1]-[3] are available.",
        "Citation [4] does not exist; only sources [1]-[3] are available.",
    ]
    assert check_citations("nothing found", 0) == []
