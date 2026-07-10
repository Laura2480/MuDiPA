"""Structural HARD constraints on SDRT discourse relations for multi-party dialogue.

Some relations are partly decidable by RULE, independent of the model — e.g. a
Question-answer_pair cannot hold when the *parent* (the question) and the *child*
(the answer) are the SAME speaker (you do not answer your own question), and a QAP
parent must actually be a question. Using these rules to mask invalid candidates
removes a class of model error / ambiguity for free ("hard deciding").

Convention: an arc is parent (p, the earlier/source unit) -> child (t, the later
unit being attached). `valid_relation(rel, p_text, t_text, p_spk, t_spk)` returns
False when the relation is structurally impossible for that arc; the scorer then
zeroes those candidates and renormalises.

These are intentionally CONSERVATIVE (only rule out clear violations) so we never
hide a legitimate label. Tune per dataset if needed.
"""
import re

_Q = re.compile(r"\?")


def is_question(text):
    """Heuristic: the utterance is interrogative (ends with / contains '?', or opens
    with a wh-/aux question word)."""
    t = (text or "").strip().lower()
    if _Q.search(t):
        return True
    return bool(re.match(r"^(who|what|where|when|why|how|which|whose|whom|"
                         r"do|does|did|is|are|was|were|can|could|will|would|"
                         r"should|shall|may|might|have|has|had|am)\b", t))


# Each rule: relation canonical-key -> predicate(p_text, t_text, p_spk, t_spk) -> bool (valid?)
# Only list relations that have a clear structural constraint; everything else is allowed.
RULES = {
    # Q-A pair: parent must be a question, answered by a DIFFERENT speaker, and the
    # child is itself an ANSWER (not a question — a question back is Clarification_q).
    "qap": lambda pt, tt, ps, ts: is_question(pt) and ps != ts and not is_question(tt),
    # Acknowledgement: you acknowledge SOMEONE ELSE's contribution; an ack is an
    # acceptance, not a question.
    "acknowledgement": lambda pt, tt, ps, ts: ps != ts and not is_question(tt),
    # Clarification question: the CHILD is a question (asking to clarify the parent),
    # by a DIFFERENT speaker.
    "clarificationq": lambda pt, tt, ps, ts: is_question(tt) and ps != ts,
    # Confirmation question: the CHILD is a question seeking confirmation.
    "confirmationq": lambda pt, tt, ps, ts: is_question(tt),
    # Question-elaboration: the CHILD is a question elaborating the parent.
    "qelab": lambda pt, tt, ps, ts: is_question(tt),
    # Correction: corrects someone else's claim → different speakers; a correction is
    # an assertion, not a question.
    "correction": lambda pt, tt, ps, ts: ps != ts and not is_question(tt),
}

# Coordinating vs subordinating SDRT relations (Asher & Lascarides 2003). This typing
# drives the Right-Frontier Constraint on LINKS (see right_frontier() below): after a
# COORDINATING relation the parent leaves the right frontier; a SUBORDINATING relation
# keeps it. Used for hard-deciding LINK candidates, not relation labels.
SUBORDINATING = {"elaboration", "explanation", "comment", "qelab", "clarificationq",
                 "confirmationq", "background", "qap", "acknowledgement", "correction"}
COORDINATING = {"narration", "continuation", "result", "parallel", "contrast",
                "alternation", "conditional", "sequence"}


from relations import canonical_relation as _canon   # single source of truth


# Relations whose CHILD is itself a question. Global typing rule: a question child can
# only take one of these; a non-question child can take none of them.
QUESTION_RELS = {"clarificationq", "confirmationq", "qelab"}


def valid_relation(rel, p_text, t_text, p_spk, t_spk):
    """True if `rel` is structurally possible for arc parent->child; False if a hard
    rule rules it out. Unknown relations are always allowed."""
    k = _canon(rel)
    # Global question-typing rule (decidable from '?'): question child ⇒ only a
    # question-relation; non-question child ⇒ never a question-relation.
    child_q = is_question(t_text)
    if child_q and k in COORDINATING:           # a question can't be Narration/Result/… of its parent
        return False
    if not child_q and k in QUESTION_RELS:
        return False
    rule = RULES.get(k)
    if rule is None:
        return True
    try:
        return bool(rule(p_text, t_text, p_spk, t_spk))
    except Exception:
        return True


def right_frontier(arcs, target):
    """SDRT Right-Frontier Constraint (Asher & Lascarides 2003): a new unit may only
    attach to a node on the RIGHT FRONTIER. In dependency form the frontier is the most
    recent attached unit plus the chain of its ancestors reachable by climbing parents;
    a SUBORDINATING incoming relation keeps the ancestor on the frontier, a COORDINATING
    one also exposes it (sibling access). Returns the SET of valid parent indices for
    `target` (units < target). EDU 0 (root) is always attachable. Conservative: if the
    structure is empty, everything < target is allowed.

    `arcs` = [{x,y,type}] already committed. Used to HARD-decide LINK candidates."""
    n_parented = {max(int(a["x"]), int(a["y"])): (min(int(a["x"]), int(a["y"])), _canon(a.get("type")))
                  for a in arcs}
    # most recent attached unit before target
    prev = max([c for c in n_parented if c < target], default=None)
    if prev is None:
        return set(range(target))          # nothing yet → no constraint
    frontier = {0}                          # root always available
    cur = prev
    seen = set()
    while cur is not None and cur not in seen:
        seen.add(cur); frontier.add(cur)
        par, rel = n_parented.get(cur, (None, None))
        if par is None:
            break
        # climb to the parent; both sub/coord parents stay reachable (coordinating also
        # exposes the parent's own frontier). Stop climbing past a coordinating chain only
        # at the root for safety (conservative = keep ancestors attachable).
        cur = par
    return frontier


def on_right_frontier(arcs, target, candidate):
    return candidate in right_frontier(arcs, target)


def filter_relation_scores(relations, p_text, t_text, p_spk, t_spk):
    """Given [{relation, score}], zero out structurally-invalid relations and
    renormalise the remaining scores to sum to 1. Returns (filtered_list, ruled_out)
    where ruled_out is the list of relation names removed by a hard rule."""
    kept, ruled = [], []
    for r in relations:
        if valid_relation(r.get("relation"), p_text, t_text, p_spk, t_spk):
            kept.append(dict(r))
        else:
            ruled.append(r.get("relation"))
    tot = sum(max(0.0, k.get("score", 0.0)) for k in kept) or 1.0
    for k in kept:
        k["score"] = max(0.0, k.get("score", 0.0)) / tot
    kept.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return kept, ruled
