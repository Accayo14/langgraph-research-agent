import asyncio

import pytest

from research_agent.config import Settings
from research_agent.schemas import Critique, Draft, Plan, Source


class FakeSearcher:
    def __init__(self, delay: float = 0.0):
        self.calls: list[str] = []
        self.delay = delay

    async def retrieve(self, query):
        self.calls.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        return [
            Source(id=f"https://wiki/{query}#{i}", title=query, url=f"https://wiki/{query}", snippet=f"{query} fact {i}")
            for i in range(2)
        ]


class FakeBrain:
    """Scripted brain: plans fixed queries, drafts from a list, critiques from a list."""

    def __init__(self, critiques=(), drafts=None, queries=("topic A", "topic B")):
        self._critiques = iter(critiques)
        self._drafts = iter(drafts) if drafts else None
        self.queries = list(queries)
        self.calls = {"plan": 0, "write": 0, "critique": 0}
        self.histories = []
        self.feedback = []

    async def plan(self, question, history=()):
        self.calls["plan"] += 1
        self.histories.append(list(history))
        return Plan(queries=self.queries)

    async def write(self, question, sources, previous_draft, feedback, history=()):
        self.calls["write"] += 1
        self.feedback.append(feedback)
        if self._drafts:
            return next(self._drafts)
        n = self.calls["write"]
        return Draft(answer=f"draft v{n} using {len(sources)} sources [1]", short_answer=f"v{n}")

    async def critique(self, question, draft, sources):
        self.calls["critique"] += 1
        return next(self._critiques)


def ok(score=9):
    return Critique(score=score, approved=True, feedback="good")


def bad(queries=()):
    return Critique(score=4, approved=False, feedback="improve", missing_info_queries=list(queries))


@pytest.fixture
def test_settings():
    return Settings(_env_file=None, model="fake:model", cache_enabled=False, checkpoint_path=None)
