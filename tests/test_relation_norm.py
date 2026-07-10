"""Regression guard for the discourse-relation normalization bug.

A relation appears in three surface forms that MUST compare equal:
  gold (STAC files):  qap, clarificationq, qelab, continuation, ...
  parser (DDPE):      Question-answer_pair, Clarification_question, Q-Elab, ...
  UI / annotator:     QAP, Clarification_Q, Q-Elab, ...

The naive `re.sub("[^a-z]","",lower())` fails to collapse the parser's long forms
of QAP and Clarification_question against gold, silently deflating any parser-vs-gold
metric. `relations.canonical_relation` must collapse all three forms for every STAC
relation. Run: pytest tests/  (or: python tests/test_relation_norm.py)
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from relations import canonical_relation  # noqa: E402

# (gold short form, DDPE/parser surface form, UI/annotator canonical form)
STAC_RELATION_FORMS = [
    ("qap",             "Question-answer_pair",   "QAP"),
    ("clarificationq",  "Clarification_question", "Clarification_Q"),
    ("qelab",           "Q-Elab",                 "Q-Elab"),
    ("comment",         "Comment",                "Comment"),
    ("acknowledgement", "Acknowledgement",        "Acknowledgement"),
    ("elaboration",     "Elaboration",            "Elaboration"),
    ("continuation",    "Continuation",           "Continuation"),
    ("result",          "Result",                 "Result"),
    ("explanation",     "Explanation",            "Explanation"),
    ("correction",      "Correction",             "Correction"),
    ("contrast",        "Contrast",               "Contrast"),
    ("parallel",        "Parallel",               "Parallel"),
    ("conditional",     "Conditional",            "Conditional"),
    ("alternation",     "Alternation",            "Alternation"),
    ("background",      "Background",             "Background"),
    ("narration",       "Narration",              "Narration"),
]


def test_all_16_relations_collapse_across_surface_forms():
    """Every STAC relation's gold / parser / UI form maps to ONE canonical key."""
    for gold, parser, ui in STAC_RELATION_FORMS:
        keys = {canonical_relation(gold), canonical_relation(parser), canonical_relation(ui)}
        assert len(keys) == 1, f"{gold}/{parser}/{ui} did not collapse: {keys}"


def test_qap_explicit():
    """The exact case the ad-hoc normalizer got wrong (QAP)."""
    assert canonical_relation("Question-answer_pair") == canonical_relation("qap")
    assert canonical_relation("QAP") == canonical_relation("qap")


def test_clarification_question_explicit():
    """The exact case the ad-hoc normalizer got wrong (Clarification_question)."""
    assert canonical_relation("Clarification_question") == canonical_relation("clarificationq")
    assert canonical_relation("Clarification_Q") == canonical_relation("clarificationq")


def test_distinct_relations_stay_distinct():
    """Different relations must NOT collapse together."""
    keys = [canonical_relation(p) for _, p, _ in STAC_RELATION_FORMS]
    assert len(set(keys)) == len(keys), "two different relations share a canonical key"


def test_the_ad_hoc_normalizer_would_fail():
    """Documents the bug: the naive normalizer disagrees with gold on QAP / Clar-Q."""
    adhoc = lambda r: re.sub(r"[^a-z]", "", str(r).lower())
    assert adhoc("Question-answer_pair") != adhoc("qap")            # the bug
    assert adhoc("Clarification_question") != adhoc("clarificationq")  # the bug
    # ...which canonical_relation fixes:
    assert canonical_relation("Question-answer_pair") == canonical_relation("qap")
    assert canonical_relation("Clarification_question") == canonical_relation("clarificationq")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS  {name}")
    print("\nAll relation-normalization regression tests passed.")
