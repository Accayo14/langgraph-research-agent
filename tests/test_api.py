from conftest import FakeBrain, FakeSearcher, bad, ok
from fastapi.testclient import TestClient

from research_agent.api import create_app


def client_for(brain, settings):
    return TestClient(create_app(brain=brain, searcher=FakeSearcher(), settings=settings))


def test_validation_errors(test_settings):
    with client_for(FakeBrain([]), test_settings) as c:
        assert c.post("/research", json={"question": "   short   "}).status_code == 422
        assert c.post("/research", json={"question": "A valid question?", "max_iterations": 9}).status_code == 422
        assert c.post("/research", json={"question": "A valid question?", "thread_id": "bad id!"}).status_code == 422


def test_research_returns_answer_and_metrics(test_settings):
    with client_for(FakeBrain([bad(), ok()]), test_settings) as c:
        body = c.post("/research", json={"question": "A valid question?"}).json()
    assert body["stop_reason"] == "approved"
    assert body["iterations"] == 2
    assert body["citations_valid"] is True
    assert body["short_answer"] == "v2"
    assert body["metrics"]["node_latency_ms"]["draft"] >= 0
    assert body["thread_id"]


def test_follow_up_and_thread_history(test_settings):
    with client_for(FakeBrain([ok(), ok()]), test_settings) as c:
        first = c.post("/research", json={"question": "Who founded SpaceX?"}).json()
        tid = first["thread_id"]
        c.post("/research", json={"question": "When was he born?", "thread_id": tid})
        turns = c.get(f"/threads/{tid}").json()["turns"]
        assert [t["question"] for t in turns] == ["Who founded SpaceX?", "When was he born?"]
        assert c.get("/threads/nope").status_code == 404


def test_stream_emits_steps_then_result(test_settings):
    with client_for(FakeBrain([ok()]), test_settings) as c:
        text = c.post("/research/stream", json={"question": "A valid question?"}).text
    # plan, 2x retrieve, select, draft, verify, critique, finalize
    assert text.count("event: step") == 8
    assert text.rstrip().split("\n")[-2] == "event: result"
