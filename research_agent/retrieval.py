import math
import re
from collections import Counter

from .schemas import Source

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by did do does for from had has have he her his how in is it its of on or "
    "she that the their they this to was were what when where which who whom why will with".split()
)
_TAIL_SECTIONS = re.compile(
    r"\n==+\s*(See also|References|External links|Notes|Further reading|Bibliography)\s*==+", re.I
)
_HEADING = re.compile(r"^==+.*==+$")


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def trim_article(text: str, max_chars: int) -> str:
    m = _TAIL_SECTIONS.search(text)
    if m:
        text = text[: m.start()]
    return text[:max_chars].rstrip()


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    chunks, cur = [], ""
    for para in (p.strip() for p in text.split("\n")):
        if not para or _HEADING.match(para):
            continue
        pieces = [para] if len(para) <= max_chars else re.split(r"(?<=[.!?])\s+", para)
        for piece in pieces:
            if cur and len(cur) + len(piece) + 1 > max_chars:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    terms = set(tokenize(query))
    toks = [tokenize(d) for d in docs]
    if not toks:
        return []
    n = len(toks)
    avg_len = sum(map(len, toks)) / n or 1
    df = Counter(t for d in toks for t in set(d))
    scores = []
    for d in toks:
        tf = Counter(d)
        s = 0.0
        for t in terms & tf.keys():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(s)
    return scores


def rank_passages(query: str, passages: list[Source], k: int = 8, per_page: int = 3) -> list[Source]:
    """Top-k passages by BM25, with at most `per_page` from the same article."""
    scores = bm25_scores(query, [f"{p.title}. {p.snippet}" for p in passages])
    order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    chosen, used = [], Counter()
    for i in order:
        p = passages[i]
        if scores[i] <= 0 or used[p.url] >= per_page:
            continue
        chosen.append(p)
        used[p.url] += 1
        if len(chosen) == k:
            break
    return chosen or passages[:k]
