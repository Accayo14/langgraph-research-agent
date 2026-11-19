import re
import string
from collections import Counter

_CITATIONS = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def normalize_answer(s: str) -> str:
    """HotpotQA / SQuAD answer normalisation."""
    s = _CITATIONS.sub(" ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        return float(p == g)
    p_toks, g_toks = p.split(), g.split()
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p_toks), common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def answer_in_text(gold: str, text: str) -> float:
    g = normalize_answer(gold)
    return float(bool(g) and f" {g} " in f" {normalize_answer(text)} ")


def answer_found(gold: str, short: str, answer: str, question: str) -> float:
    """Lenient accuracy: the gold answer appears in the short or long answer.

    If the gold answer is already in the question (e.g. "which of A or B..."), any answer that
    repeats the question would match, so only the short answer counts.
    """
    if answer_in_text(gold, question):
        return answer_in_text(gold, short)
    return max(answer_in_text(gold, short), answer_in_text(gold, answer))


def support_recall(gold_titles: list[str], source_titles: list[str]) -> float:
    """Fraction of the gold supporting Wikipedia pages that made it into the selected sources."""
    gold = {t.lower() for t in gold_titles}
    if not gold:
        return 1.0
    return len(gold & {t.lower() for t in source_titles}) / len(gold)
