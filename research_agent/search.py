import asyncio
import json
from typing import Protocol

import httpx

from .cache import Cache, make_key
from .retrieval import chunk_text, trim_article
from .schemas import Source

WIKI_API = "https://en.wikipedia.org/w/api.php"
RETRY_STATUSES = {429, 500, 502, 503, 504}
USER_AGENT = "langgraph-research-agent/0.2 (https://github.com/Accayo14/langgraph-research-agent)"


class Searcher(Protocol):
    async def retrieve(self, query: str) -> list[Source]: ...


class WikipediaSearcher:
    """Searches Wikipedia, fetches full articles and splits them into passages."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        pages_per_query: int = 3,
        max_page_chars: int = 20_000,
        cache: Cache | None = None,
        max_concurrency: int = 4,
        retries: int = 3,
    ):
        self._client = client
        self._k = pages_per_query
        self._max_chars = max_page_chars
        self._cache = cache
        # wikipedia starts returning 429s if we hit it with too many requests at once
        self._sem = asyncio.Semaphore(max_concurrency)
        self._retries = retries
        self.failed_requests = 0

    async def _fetch(self, params: dict) -> dict | None:
        for attempt in range(self._retries + 1):
            async with self._sem:
                try:
                    resp = await self._client.get(WIKI_API, params=params)
                except httpx.HTTPError:
                    resp = None
            if resp is not None and resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return None
            retryable = resp is None or resp.status_code in RETRY_STATUSES
            if not retryable or attempt == self._retries:
                return None
            wait = 0.5 * 2**attempt
            if resp is not None and resp.headers.get("retry-after", "").isdigit():
                wait = max(wait, float(resp.headers["retry-after"]))
            await asyncio.sleep(min(wait, 10))
        return None

    async def _get(self, params: dict) -> dict:
        params = {"format": "json", "formatversion": 2, **params}
        key = make_key("wiki", params)
        if self._cache and (hit := await self._cache.get(key)) is not None:
            return json.loads(hit)
        data = await self._fetch(params)
        if data is None:
            self.failed_requests += 1
            return {}
        if self._cache:
            await self._cache.set(key, json.dumps(data))
        return data

    async def search_titles(self, query: str) -> list[str]:
        data = await self._get({"action": "query", "list": "search", "srsearch": query, "srlimit": self._k, "srprop": ""})
        return [r["title"] for r in data.get("query", {}).get("search", [])]

    async def fetch_page(self, title: str) -> list[Source]:
        data = await self._get(
            {"action": "query", "prop": "extracts|info", "explaintext": 1, "inprop": "url", "redirects": 1, "titles": title}
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or not pages[0].get("extract"):
            return []
        p = pages[0]
        chunks = chunk_text(trim_article(p["extract"], self._max_chars))
        return [Source(id=f"{p['fullurl']}#{i}", title=p["title"], url=p["fullurl"], snippet=c) for i, c in enumerate(chunks)]

    async def retrieve(self, query: str) -> list[Source]:
        titles = await self.search_titles(query)
        pages = await asyncio.gather(*(self.fetch_page(t) for t in titles))
        return [s for page in pages for s in page]


def make_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": USER_AGENT},
    )
