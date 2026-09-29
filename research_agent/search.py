import asyncio
from typing import Protocol

import httpx

from .schemas import Source

WIKI_API = "https://en.wikipedia.org/w/api.php"


class Searcher(Protocol):
    async def search_many(self, queries: list[str]) -> list[Source]: ...


class WikipediaSearcher:
    def __init__(self, client: httpx.AsyncClient, results_per_query: int = 2):
        self._client = client
        self._k = results_per_query

    async def search(self, query: str) -> list[Source]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": self._k,
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "exchars": 1200,
            "inprop": "url",
        }
        try:
            resp = await self._client.get(WIKI_API, params=params)
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        pages = resp.json().get("query", {}).get("pages", {}).values()
        pages = sorted(pages, key=lambda p: p.get("index", 0))
        return [
            Source(title=p["title"], url=p["fullurl"], snippet=p.get("extract", "").strip())
            for p in pages
            if p.get("extract")
        ]

    async def search_many(self, queries: list[str]) -> list[Source]:
        results = await asyncio.gather(*(self.search(q) for q in queries))
        return [s for batch in results for s in batch]


def make_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=10.0,
        headers={"User-Agent": "langgraph-research-agent/0.1 (educational project)"},
    )
