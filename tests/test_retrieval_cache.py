import asyncio

import httpx
from conftest import FakeBrain
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from research_agent.cache import Cache, CachedBrain
from research_agent.metrics import RunMetrics
from research_agent.retrieval import bm25_scores, chunk_text, rank_passages, tokenize, trim_article
from research_agent.schemas import Source
from research_agent.search import WikipediaSearcher


def src(title, text, i=0):
    return Source(id=f"{title}#{i}", title=title, url=f"https://wiki/{title}", snippet=text)


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("Who was the CEO of Tesla, Inc.?") == ["ceo", "tesla", "inc"]


def test_trim_article_cuts_reference_sections():
    text = "Intro.\n\n== History ==\nStuff.\n\n== References ==\nJunk"
    assert trim_article(text, 1000) == "Intro.\n\n== History ==\nStuff."


def test_chunk_text_respects_size_and_skips_headings():
    text = "== Heading ==\n" + "\n".join(["word " * 30] * 10)
    chunks = chunk_text(text, max_chars=400)
    assert all(len(c) <= 400 for c in chunks)
    assert not any("==" in c for c in chunks)
    assert len(chunks) > 1


def test_bm25_prefers_matching_documents():
    scores = bm25_scores("Ed Wood nationality", ["Ed Wood was an American filmmaker.", "Bananas are yellow."])
    assert scores[0] > scores[1] == 0


def test_rank_passages_caps_per_page_and_drops_irrelevant():
    passages = [src("Ed Wood", f"Ed Wood film {i}", i) for i in range(5)] + [src("Villa", "a country house")]
    chosen = rank_passages("Ed Wood film", passages, k=8, per_page=3)
    assert len(chosen) == 3
    assert all(p.title == "Ed Wood" for p in chosen)


def test_cached_brain_skips_repeat_calls(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    inner = FakeBrain([])
    brain = CachedBrain(inner, cache, namespace="test")
    first = asyncio.run(brain.plan("Who founded SpaceX?"))
    second = asyncio.run(brain.plan("Who founded SpaceX?"))
    assert first == second
    assert inner.calls["plan"] == 1
    assert (cache.hits, cache.misses) == (1, 1)
    cache.close()

    reopened = Cache(str(tmp_path / "c.sqlite"))  # persists across restarts
    asyncio.run(CachedBrain(inner, reopened, namespace="test").plan("Who founded SpaceX?"))
    assert inner.calls["plan"] == 1
    reopened.close()


def fake_wikipedia(request: httpx.Request) -> httpx.Response:
    params = request.url.params
    if params.get("list") == "search":
        return httpx.Response(200, json={"query": {"search": [{"title": "Ed Wood"}, {"title": "Missing"}]}})
    if params["titles"] == "Ed Wood":
        page = {"title": "Ed Wood", "fullurl": "https://en.wikipedia.org/wiki/Ed_Wood",
                "extract": "Ed Wood was an American filmmaker.\n\n== References ==\nx"}
        return httpx.Response(200, json={"query": {"pages": [page]}})
    return httpx.Response(500)


def test_wikipedia_searcher_chunks_pages_and_caches():
    calls = []

    def handler(request):
        calls.append(request)
        return fake_wikipedia(request)

    async def go():
        cache = Cache()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            s = WikipediaSearcher(client, cache=cache, retries=0)
            first = await s.retrieve("Ed Wood")
            second = await s.retrieve("Ed Wood")
        cache.close()
        return first, second, s.failed_requests

    first, second, failed = asyncio.run(go())
    assert [p.snippet for p in first] == ["Ed Wood was an American filmmaker."]
    assert first[0].id == "https://en.wikipedia.org/wiki/Ed_Wood#0"
    assert first == second
    assert len(calls) == 4  # search + 2 pages, then only the failed page is tried again
    assert failed == 2


def test_wikipedia_searcher_retries_rate_limits():
    responses = iter([httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json={"query": {"search": []}})])

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: next(responses))) as client:
            s = WikipediaSearcher(client, retries=2)
            return await s.search_titles("anything"), s.failed_requests

    assert asyncio.run(go()) == ([], 0)


def test_run_metrics_counts_tokens_models_and_errors():
    m = RunMetrics()
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "vendor/model"},
    )
    m.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    m.on_llm_error(RuntimeError("429"))
    s = m.summary()
    assert (s.llm_calls, s.failed_llm_calls, s.prompt_tokens, s.completion_tokens) == (1, 1, 10, 5)
    assert s.models_used == {"vendor/model": 1}
