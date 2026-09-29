import asyncio

from fastapi.testclient import TestClient

from research_agent.api import create_app
from research_agent.graph import build_graph
from research_agent.schemas import Critique, Plan, Source


class FakeSearcher:
    def __init__(self):
        self.calls: list[list[str]] = []

    async def search_many(self, queries):
        self.calls.append(list(queries))
        return [Source(title=q, url=f"https://wiki/{q}", snippet=f"facts about {q}") for q in queries]


class FakeBrain:
    def __init__(self, critiques: list[Critique]):
        self._critiques = iter(critiques)
        self.write_calls = 0

    async def plan(self, question):
        return Plan(queries=["topic A", "topic B"])

    async def write(self, question, sources, previous_draft, feedback):
        self.write_calls += 1
        return f"draft v{self.write_calls} using {len(sources)} sources"

    async def critique(self, question, draft, sources):
        return next(self._critiques)


def run(brain, searcher, max_iterations=3):
    graph = build_graph(brain, searcher)
    return asyncio.run(graph.ainvoke({"question": "What is X and why?", "max_iterations": max_iterations}))


def ok(score=9):
    return Critique(score=score, approved=True, feedback="good")


def bad(queries=()):
    return Critique(score=4, approved=False, feedback="improve", missing_info_queries=list(queries))


def test_approved_first_time_takes_straight_path():
    s = FakeSearcher()
    out = run(FakeBrain([ok()]), s)
    assert out["stop_reason"] == "approved"
    assert out["iteration"] == 1
    assert len(s.calls) == 1


def test_revision_loop_goes_back_to_draft():
    out = run(FakeBrain([bad(), ok()]), FakeSearcher())
    assert out["iteration"] == 2
    assert out["draft"] == "draft v2 using 2 sources"
    assert out["stop_reason"] == "approved"


def test_research_loop_goes_back_to_search_and_merges_sources():
    s = FakeSearcher()
    out = run(FakeBrain([bad(["topic C"]), ok()]), s)
    assert s.calls == [["topic A", "topic B"], ["topic C"]]
    assert [src.title for src in out["sources"]] == ["topic A", "topic B", "topic C"]


def test_repeated_queries_are_not_searched_again():
    s = FakeSearcher()
    run(FakeBrain([bad(["Topic A"]), ok()]), s)
    assert len(s.calls) == 1


def test_loop_stops_at_max_iterations():
    brain = FakeBrain([bad()] * 10)
    out = run(brain, FakeSearcher(), max_iterations=3)
    assert out["stop_reason"] == "max_iterations"
    assert brain.write_calls == 3


def test_api_validation_and_response():
    app = create_app(brain=FakeBrain([bad(), ok()]), searcher=FakeSearcher())
    with TestClient(app) as client:
        assert client.post("/research", json={"question": "   short   "}).status_code == 422
        assert client.post("/research", json={"question": "A valid question?", "max_iterations": 9}).status_code == 422

        r = client.post("/research", json={"question": "A valid question?", "max_iterations": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["stop_reason"] == "approved"
        assert body["iterations"] == 2
        assert len(body["sources"]) == 2


def test_api_stream_emits_steps_then_result():
    app = create_app(brain=FakeBrain([ok()]), searcher=FakeSearcher())
    with TestClient(app) as client:
        text = client.post("/research/stream", json={"question": "A valid question?"}).text
    assert text.count("event: step") == 5
    assert text.rstrip().split("\n")[-2] == "event: result"
