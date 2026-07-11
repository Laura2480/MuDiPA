"""SDRT relation set, definitions and normalization for MuDiPA (extracted from DSDP)."""
import re

# ── Canonical COMPARISON key ─────────────────────────────────────────────────
# A relation appears in three surface forms across the codebase and they must
# compare equal:
#   gold (STAC files):   qap, clarificationq, qelab, continuation, ...
#   parser (DDPE):       Question-answer_pair, Clarification_question, Q-Elab, ...
#   UI / annotator:      QAP, Clarification_Q, Q-Elab, ...
# The naive `re.sub("[^a-z]","",lower())` collapses 14/16 relations but NOT the
# parser's long forms of QAP and Clarification_question (-> "questionanswerpair"
# / "clarificationquestion", which never equal gold "qap" / "clarificationq").
# `canonical_relation` fixes exactly those, so it is identity-correct for the
# other relations. Use it for EVERY relation equality / F1 / accuracy check.
_CANONICAL_KEY_ALIASES = {
    "questionanswerpair":    "qap",             # DDPE Question-answer_pair
    "questionanswer":        "qap",
    "clarificationquestion": "clarificationq",  # DDPE Clarification_question
    "clarification":         "clarificationq",
    "confirmationquestion":  "confirmationq",   # DDPE Confirmation_question (msdc)
    "questionelaboration":   "qelab",           # long form of Q-Elab
    "acknowledgment":        "acknowledgement",  # US spelling
}


def canonical_relation(rel):
    """Return the single comparison key for a relation, collapsing gold / parser /
    UI surface forms. For COMPARISON only (not display). Empty/None -> ""."""
    key = re.sub(r"[^a-z]", "", str(rel or "").lower())
    return _CANONICAL_KEY_ALIASES.get(key, key)


RELATION_NORMALIZE = {
    "Question_answer_pair": "QAP", "QAP": "QAP", "Question-answer_pair": "QAP",
    "Comment": "Comment",
    "Clarification_question": "Clarification_Q", "Clarification_Q": "Clarification_Q",
    "Elaboration": "Elaboration",
    "Continuation": "Continuation",
    "Q_Elab": "Q-Elab", "Q-Elab": "Q-Elab",
    "Explanation": "Explanation", "Result": "Result",
    "Contrast": "Contrast", "Correction": "Correction",
    "Conditional": "Conditional", "Alternation": "Alternation",
    "Background": "Background", "Narration": "Narration",
    "Parallel": "Parallel", "Acknowledgement": "Acknowledgement",
    "Confirmation_question": "Confirmation_Q", "Confirmation_Q": "Confirmation_Q",
    "Sequence": "Sequence",
}


def normalize_relation(raw_type):
    """Map any dataset relation label to canonical form."""
    return RELATION_NORMALIZE.get(raw_type, raw_type)


_REL_DEF = {
    "QAP":              "X is a question, Y is the answer to X",
    "Comment":          "Y gives an opinion or reaction to X",
    "Clarification_Q":  "Y is a question that asks for clarification OF X (X was said first; Y follows and asks about X)",
    "Elaboration":      "Y adds more detail or information to X",
    "Continuation":     "Y continues the same speech act as X (often same speaker, same topic)",
    "Q-Elab":           "Y is a follow-up question that elaborates on X",
    "Explanation":      "Y gives the reason or justification for X",
    "Result":           "Y is a consequence or result of X",
    "Contrast":         "Y opposes or contrasts with X",
    "Correction":       "Y corrects or redirects X",
    "Conditional":      "Y states a condition for X",
    "Alternation":      "Y presents an alternative to X",
    "Background":       "Y provides background context for X",
    "Narration":        "Y describes the next event after X (sequential narrative)",
    "Parallel":         "Y makes a parallel statement to X",
    "Acknowledgement":  "Y acknowledges or confirms X (e.g. 'ok', 'sure', 'yeah')",
    "Confirmation_Q":   "Y is a yes/no question seeking confirmation of X",
    "Sequence":         "Y is the next step in a sequence initiated by X",
}

DATASET_RELATIONS = {
    "stac":    ["QAP", "Comment", "Clarification_Q", "Elaboration", "Continuation",
                "Q-Elab", "Explanation", "Result", "Contrast", "Correction",
                "Conditional", "Alternation", "Background", "Narration", "Parallel",
                "Acknowledgement"],
    "molweni": ["QAP", "Comment", "Clarification_Q", "Elaboration", "Continuation",
                "Q-Elab", "Explanation", "Result", "Contrast", "Correction",
                "Conditional", "Alternation", "Background", "Narration", "Parallel",
                "Acknowledgement"],
    "msdc":    ["QAP", "Comment", "Clarification_Q", "Elaboration", "Continuation",
                "Q-Elab", "Explanation", "Result", "Contrast", "Correction",
                "Conditional", "Alternation", "Narration", "Acknowledgement",
                "Confirmation_Q", "Sequence"],
}

_ACTIVE_DATASET = "stac"


def build_relation_defs(dataset=None):
    """Build the RELATION_DEFS text block for a specific dataset."""
    ds = dataset or _ACTIVE_DATASET
    rels = DATASET_RELATIONS.get(ds, DATASET_RELATIONS["stac"])
    lines = ["Possible discourse relations between two utterances (X -> Y):",
             "X is the earlier utterance, Y is the source/antecedent and Y is the dependent."]
    for r in rels:
        lines.append(f"- {r}: {_REL_DEF[r]}")
    return "\n".join(lines)


RELATION_DEFS = build_relation_defs("stac")


def relation_def(rel):
    """One-line codebook definition for a SINGLE relation (case-insensitive), or ''.
    The reasoning engine judges one proposed relation against ITS OWN definition,
    without being shown the rest of the inventory."""
    if not rel:
        return ""
    r = RELATION_NORMALIZE.get(rel, rel)
    if r in _REL_DEF:
        return f"{r}: {_REL_DEF[r]}"
    for k, v in _REL_DEF.items():
        if k.lower() == str(rel).lower():
            return f"{k}: {v}"
    return ""

CANONICAL_RELATIONS_EXT = {
    "QAP", "Elaboration", "Continuation", "Acknowledgement", "Q-Elab",
    "Comment", "Contrast", "Alternation", "Parallel", "Correction",
    "Result", "Explanation", "Narration", "Background", "Conditional",
    "Clarification_Q", "Confirmation_Q", "Sequence",
}

_REL_ALIASES_EXT = {
    "qap": "QAP", "question-answer pair": "QAP",
    "elaboration": "Elaboration", "continuation": "Continuation",
    "acknowledgement": "Acknowledgement", "ack": "Acknowledgement",
    "q-elab": "Q-Elab", "comment": "Comment", "contrast": "Contrast",
    "parallel": "Parallel", "correction": "Correction", "result": "Result",
    "explanation": "Explanation", "narration": "Narration",
    "background": "Background", "conditional": "Conditional",
    "clarification_q": "Clarification_Q", "clarification": "Clarification_Q",
    "sequence": "Sequence", "answering": "QAP", "response": "QAP",
}


def normalize_relation_ext(rel):
    if not rel or rel.lower() in ("null", "none", "unrelated"):
        return None
    if rel in CANONICAL_RELATIONS_EXT:
        return rel
    return _REL_ALIASES_EXT.get(rel.lower(), rel.title())
