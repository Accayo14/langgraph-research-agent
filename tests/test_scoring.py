from evals.run import rescore, summarize
from evals.scoring import answer_found, exact_match, f1, normalize_answer, support_recall


def test_normalize_strips_case_articles_punctuation_and_citations():
    assert normalize_answer("The  Dutch [1]!") == "dutch"


def test_exact_match_and_f1():
    assert exact_match("Kevin Spacey", "kevin spacey") == 1.0
    assert f1("Kevin Spacey Fowler", "Kevin Spacey") == 0.8
    assert f1("no", "yes") == 0.0


def test_answer_found_ignores_answers_copied_from_the_question():
    q = "Between Greyia and Calibanus, which genus contains more species?"
    refusal = "The sources say nothing about Greyia or Calibanus."
    assert answer_found("Greyia", "Unable to determine", refusal, q) == 0.0
    assert answer_found("Greyia", "Greyia", refusal, q) == 1.0
    assert answer_found("1755", "unknown", "It was founded in 1755 [2].", "When was it founded?") == 1.0


def test_support_recall():
    assert support_recall(["Ed Wood", "Scott Derrickson"], ["ed wood", "Villa"]) == 0.5


def test_rescore_and_summarize():
    rec = {"id": "1", "variant": "agent", "question": "Who?", "gold": "Ann", "short_answer": "Ann", "answer": "Ann [1]",
           "support_recall": 1.0, "citations_valid": True, "iterations": 1, "llm_calls": 3, "tokens": 10,
           "latency_s": 2.0, "failed_llm_calls": 1}
    rows = summarize([rescore(rec), {"id": "2", "variant": "agent", "error": "boom"}])
    assert rows[0]["exact_match"] == 1.0
    assert rows[0]["errors"] == 1
