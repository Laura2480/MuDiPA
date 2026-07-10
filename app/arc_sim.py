"""
Discourse-functional arc similarity for confusion-cell retrieval.

The confusion cell (gold, pred) is a HARD facet: retrieval happens only within
one relation-pair. Within a cell, examples are ranked by how much their
*discourse configuration* resembles the query arc — NOT by topic. For short
game chat, topic (wheat/sheep/clay) is orthogonal to the discourse relation;
the relation signal lives in speaker structure, attachment distance, the
dialogue act of the target, and the cue markers. So similarity is dominated by
those, with raw-token overlap kept as a small tie-break.

Dependency-free (regex + set ops) so it runs inside the Flask process without
loading an embedding model.
"""
import re

# SDRT cue-marker families for multi-party chat. Presence of a family in the
# target utterance is the strongest lexical signal for the relation.
_CUE = {
    "ack":   {"ok", "okay", "k", "kk", "yeah", "yea", "yep", "yes", "sure",
              "right", "np", "thanks", "thx", "ty", "alright", "fine", "gotcha",
              "agreed", "deal", "true", "cool", "sounds"},
    "eval":  {"nice", "wow", "lol", "lmao", "haha", "hah", "damn", "ugh",
              "great", "good", "bad", "crazy", "funny", "oh", "ah", "hmm",
              "sucks", "awesome", "cute", "yay", "yey", "argh", "grr"},
    "cont":  {"and", "also", "too", "plus", "then", "still", "aswell"},
    "cause": {"because", "cos", "cause", "cuz", "since", "so", "therefore", "thus"},
    "contr": {"but", "however", "actually", "no", "nope", "wait", "though",
              "instead", "although", "well"},
    "cond":  {"if", "unless", "or", "either", "otherwise"},
}
_QWORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}
_AUX    = {"do", "does", "did", "can", "could", "would", "will", "is", "are",
           "am", "was", "were", "should", "have", "has", "any", "anyone", "u"}


def _toks(s):
    return re.findall(r"[a-z']+|\?", (s or "").lower())


def _is_question(text):
    t = (text or "").strip()
    if t.endswith("?"):
        return True
    tk = _toks(t)
    return bool(tk) and (tk[0] in _QWORDS or tk[0] in _AUX)


def _cue_cats(tokens):
    tset = set(tokens)
    cats = {c for c, ws in _CUE.items() if tset & ws}
    if (tset & _QWORDS) or "?" in tset:
        cats.add("qword")
    return cats


def arc_signature(src_speaker, src_text, tgt_speaker, tgt_text, distance):
    """Discourse-functional fingerprint of an arc (source -> target)."""
    st, tt = _toks(src_text), _toks(tgt_text)
    return {
        "same_speaker": (src_speaker or "").strip().lower()
                        == (tgt_speaker or "").strip().lower(),
        "dist_bucket": 0 if distance <= 1 else (1 if distance <= 3 else 2),
        "tgt_q": _is_question(tgt_text),
        "src_q": _is_question(src_text),
        "tgt_short": len(tt) <= 2,
        "tgt_cats": _cue_cats(tt),
        "src_cats": _cue_cats(st),
        "tgt_tok": set(tt),
        "src_tok": set(st),
    }


def _jacc(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def arc_similarity(a, b):
    """Weighted discourse similarity in [0,1]. Structure + cue markers dominate
    (0.94); raw-token/topic overlap is a deliberate 0.06 tie-break."""
    s = 0.30 * (a["same_speaker"] == b["same_speaker"])
    s += 0.15 * (a["dist_bucket"] == b["dist_bucket"])
    s += 0.12 * (a["tgt_q"] == b["tgt_q"])
    s += 0.06 * (a["src_q"] == b["src_q"])
    s += 0.07 * (a["tgt_short"] == b["tgt_short"])
    s += 0.18 * _jacc(a["tgt_cats"], b["tgt_cats"])   # target cue family = relation signal
    s += 0.06 * _jacc(a["src_cats"], b["src_cats"])
    s += 0.04 * _jacc(a["tgt_tok"], b["tgt_tok"])     # topic tie-break, small on purpose
    s += 0.02 * _jacc(a["src_tok"], b["src_tok"])
    return round(s, 4)


def signature_of_example(ex):
    """Signature for an index example dict (has src/tgt speaker+text+distance)."""
    return arc_signature(ex["src_speaker"], ex["src_text"],
                         ex["tgt_speaker"], ex["tgt_text"],
                         ex.get("distance", abs(ex["tgt"] - ex["src"])))
