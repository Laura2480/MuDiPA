"""Cross-framework discourse-relation glossary for the MuDiPA codebook.

Definitions follow the ORIGINAL taxonomies:
  - PDTB : PDTB 3.0 sense hierarchy (Webber et al., 2019 annotation manual)
  - RST  : Mann & Thompson (1988) + RST-DT (Carlson & Marcu)
  - eRST : GUM enhanced-RST relation set (Zeldes et al.)
  - DEP  : SciDTB discourse-dependency relations (Yang & Li, 2018)
  - ISO  : ISO 24617-8 DR-core
  - SDRT : STAC/Molweni SDRT relations (Asher et al.)

PDTB and RST/eRST labels are `.`/`-`-structured, so instead of enumerating the
hundreds of (multilingual, multi-sense) leaves we define the COMPONENTS once and
compose a definition for any label. `define(label, framework)` returns a one-line
codebook gloss, or "" when unknown (the UI then shows the raw label). Non-English
RST labels (Basque/Spanish/…) fall back to their English cognate when obvious, else "".
"""

# ── PDTB 3.0 ────────────────────────────────────────────────────────────────
# keyed by lowercased sense (full leaf, or coarser); compose() picks the most specific.
_PDTB = {
    "temporal": "the two situations are related in time.",
    "temporal.asynchronous": "one situation precedes the other in time.",
    "temporal.asynchronous.precedence": "Arg1 happens before Arg2 (then/after).",
    "temporal.asynchronous.succession": "Arg1 happens after Arg2 (earlier/before).",
    "temporal.synchronous": "the two situations overlap in time (while/when).",
    "temporal.synchrony": "the two situations overlap in time (while/when).",
    "contingency": "one argument causally influences or enables the other.",
    "causation": "one argument causes the other.",
    "contingency.cause": "a cause–effect link between the arguments.",
    "contingency.cause.reason": "Arg2 is the reason/cause for Arg1 (because).",
    "contingency.cause.result": "Arg2 is the result/effect of Arg1 (so/therefore).",
    "contingency.cause+belief.reason+belief": "Arg2 is the evidence for believing Arg1.",
    "contingency.cause+belief.result+belief": "Arg1 is the evidence for believing Arg2.",
    "contingency.cause+speechact.reason+speechact": "Arg2 is the reason justifying the speech act in Arg1.",
    "contingency.cause+speechact.result+speechact": "the speech act in Arg2 follows as a result of Arg1.",
    "contingency.condition": "one argument is the condition for the other (if).",
    "contingency.condition.arg1-as-cond": "Arg1 is the condition, Arg2 the consequent (if Arg1, Arg2).",
    "contingency.condition.arg2-as-cond": "Arg2 is the condition, Arg1 the consequent.",
    "contingency.condition+speechact": "a condition placed on the speech act.",
    "contingency.negative-condition.arg1-as-negcond": "Arg1 is a negative condition (unless).",
    "contingency.negative-condition.arg2-as-negcond": "Arg2 is a negative condition (unless).",
    "contingency.negative-cause.negresult": "the absence of the cause yields the negated result.",
    "contingency.purpose": "one argument is the goal, the other the means/action.",
    "contingency.purpose.arg1-as-goal": "Arg1 is the goal, Arg2 the means/action.",
    "contingency.purpose.arg2-as-goal": "Arg2 is the goal, Arg1 the means/action (in order to).",
    "contingency.goal": "one argument states the goal of the other.",
    "comparison": "the arguments are compared for similarity or difference.",
    "comparison.concession": "one argument denies an expectation raised by the other (although).",
    "comparison.concession.arg1-as-denier": "Arg1 denies an expectation raised by Arg2 (although).",
    "comparison.concession.arg2-as-denier": "Arg2 denies an expectation raised by Arg1 (although/but).",
    "comparison.contrast": "Arg1 and Arg2 differ on shared properties (but/whereas).",
    "comparison.similarity": "Arg1 and Arg2 share a common property (similarly).",
    "comparison.degree": "the arguments are compared by degree.",
    "expansion": "one argument expands on the other (adds, specifies, exemplifies, reformulates).",
    "expansion.conjunction": "Arg2 adds to Arg1; both hold and are jointly relevant (and/also).",
    "expansion.disjunction": "Arg1 and Arg2 are alternatives (or).",
    "expansion.equivalence": "Arg2 restates the same situation as Arg1 in other words.",
    "expansion.instantiation": "one argument gives an example/instance of the other.",
    "expansion.instantiation.arg1-as-instance": "Arg1 is an example/instance of Arg2.",
    "expansion.instantiation.arg2-as-instance": "Arg2 is an example/instance of Arg1 (for example).",
    "expansion.level-of-detail.arg1-as-detail": "Arg1 gives Arg2 in more detail.",
    "expansion.level-of-detail.arg2-as-detail": "Arg2 gives Arg1 in more detail (specifically).",
    "expansion.level-of-detail": "one argument gives the other in more/less detail.",
    "expansion.manner.arg1-as-manner": "Arg1 gives the manner/means of Arg2.",
    "expansion.manner.arg2-as-manner": "Arg2 gives the manner/means of Arg1 (by).",
    "expansion.manner": "one argument gives the manner/means of the other.",
    "expansion.substitution.arg1-as-subst": "Arg1 is chosen in place of Arg2.",
    "expansion.substitution.arg2-as-subst": "Arg2 is chosen instead of Arg1 (instead).",
    "expansion.substitution": "one argument is chosen instead of the other.",
    "expansion.exception.arg1-as-except": "Arg1 is an exception to Arg2.",
    "expansion.exception.arg2-as-except": "Arg2 is an exception to Arg1 (except).",
    "expansion.exception.arg2-as-excpt": "Arg2 is an exception to Arg1 (except).",
    "expansion.exception": "one argument is an exception to the other.",
    "expansion.correction": "Arg2 corrects Arg1.",
    "expansion.restatement": "Arg2 restates Arg1.",
    "expansion.restatement.equivalence": "Arg2 restates Arg1 as an equivalent.",
    "expansion.restatement.specification": "Arg2 restates Arg1 more specifically.",
    "expansion.progression.arg2-as-progr": "Arg2 continues a progression begun in Arg1.",
    "expansion.alternative": "Arg2 is an alternative to Arg1.",
    "hypophora": "Arg1 poses a question that Arg2 answers.",
    "qap.hypophora": "Arg1 poses a question that Arg2 answers.",
    "narrative-response": "Arg2 answers/responds to a question or issue raised in Arg1.",
    "reformulation": "Arg2 reformulates Arg1.",
    "repetition": "Arg2 repeats Arg1.",
    "progression": "the arguments form a progression.",
    "purpose": "one argument states the purpose/goal of the other.",
    "alternative": "the arguments are alternatives.",
    "conjunction": "the arguments both hold and are jointly relevant (and).",
    "conditional": "one argument is a condition for the other (if).",
    "contrast": "the arguments differ (but).",
    "concession": "one argument denies an expectation raised by the other (although).",
    "comparison.concession+speechact.arg2-as-denier+speechact": "Arg2 concedes an expectation raised by the speech act in Arg1.",
}


def _pdtb(label):
    out = []
    for sense in str(label).lower().split(";"):     # multi-sense: A;B
        sense = sense.strip()
        d = _PDTB.get(sense)
        if not d:                                    # back off to coarser sense
            toks = sense.split(".")
            while toks and not d:
                d = _PDTB.get(".".join(toks))
                toks = toks[:-1]
        if d and d not in out:
            out.append(d)
    return " + also: ".join(out)


# ── RST (Mann&Thompson / RST-DT) + eRST (GUM) ───────────────────────────────
_RST = {
    # eRST class-subtype leaves
    "elaboration-additional": "Arg2 adds further information about Arg1.",
    "elaboration-attribute": "Arg2 specifies an attribute/property of something in Arg1.",
    "attribution-positive": "Arg2 attributes Arg1 to a source (says/claims/thinks).",
    "attribution-negative": "the source does NOT make the attributed statement.",
    "joint-list": "Arg2 is another item in a list with Arg1.",
    "joint-sequence": "Arg2 is the next event in a sequence after Arg1.",
    "joint-disjunction": "Arg1 and Arg2 are alternatives (or).",
    "joint-other": "Arg2 is coordinated with Arg1 without a more specific relation.",
    "causal-cause": "Arg2 is the cause of Arg1.",
    "causal-result": "Arg2 is the result of Arg1.",
    "context-background": "Arg2 gives background needed to understand Arg1.",
    "context-circumstance": "Arg2 gives the circumstance under which Arg1 holds.",
    "adversative-concession": "Arg2 concedes a point that might tell against Arg1 (although).",
    "adversative-contrast": "Arg1 and Arg2 contrast.",
    "adversative-antithesis": "Arg2 is the rejected antithesis; Arg1 is favoured.",
    "mode-manner": "Arg2 gives the manner in which Arg1 is done.",
    "mode-means": "Arg2 gives the means by which Arg1 is done.",
    "purpose-goal": "Arg2 is the goal/purpose of the action in Arg1.",
    "purpose-attribute": "Arg2 is a purpose attributed to something in Arg1.",
    "restatement-partial": "Arg2 partially restates Arg1.",
    "restatement-repetition": "Arg2 repeats Arg1.",
    "explanation-evidence": "Arg2 gives evidence for the claim in Arg1.",
    "explanation-justify": "Arg2 justifies the writer's putting forward Arg1.",
    "explanation-motivation": "Arg2 motivates the reader to act on Arg1.",
    "evaluation-comment": "Arg2 comments on or evaluates Arg1.",
    "organization-heading": "Arg1 is a heading/title for Arg2.",
    "organization-preparation": "Arg1 prepares the reader for Arg2.",
    "organization-phatic": "a phatic / discourse-management unit.",
    "topic-question": "Arg1 raises a question answered by Arg2.",
    "topic-solutionhood": "Arg2 is a solution to a problem/question in Arg1.",
    "contingency-condition": "Arg2 is a condition for Arg1 (if).",
    "same-unit": "the two spans are parts of one interrupted unit.",
    # RST-DT / classic leaves and class heads
    "elaboration": "Arg2 adds more detail/information about Arg1.",
    "e-elaboration": "Arg2 elaborates an entity mentioned in Arg1.",
    "joint": "the units are coordinated with no strong asymmetry.",
    "list": "the units form a list.",
    "sequence": "the units are in temporal/logical sequence.",
    "causal": "a cause–result link between the units.",
    "cause": "Arg2 is the cause of Arg1.",
    "result": "Arg2 is the result of Arg1.",
    "cause-result": "a cause–result link between the units.",
    "cause-effect": "a cause–effect link between the units.",
    "consequence": "one unit is the consequence of the other.",
    "context": "one unit sets the temporal/situational context for the other.",
    "background": "Arg2 gives background needed to understand Arg1.",
    "circumstance": "Arg2 gives the circumstance under which Arg1 holds.",
    "adversative": "the units contrast or one concedes to the other.",
    "concession": "Arg2 concedes a point that might tell against Arg1 (although).",
    "contrast": "the units contrast on shared properties.",
    "comparison": "the units are compared.",
    "antithesis": "Arg2 is the rejected antithesis; Arg1 is favoured.",
    "attribution": "one unit attributes a statement/attitude to a source.",
    "explanation": "one unit explains, justifies, or gives evidence/motivation for the other.",
    "evidence": "Arg2 provides evidence for the claim in Arg1.",
    "justify": "Arg2 justifies the writer's putting forward Arg1.",
    "motivation": "Arg2 motivates the reader to act on Arg1.",
    "reason": "Arg2 gives the reason for Arg1.",
    "mode": "one unit gives the manner or means of the other.",
    "manner": "Arg2 gives the manner in which Arg1 is done.",
    "means": "Arg2 gives the means by which Arg1 is done.",
    "manner-means": "Arg2 gives the manner/means of Arg1.",
    "purpose": "Arg2 is the goal/purpose of the action in Arg1.",
    "enablement": "Arg2 enables the action in Arg1.",
    "restatement": "Arg2 restates Arg1.",
    "reformulation": "Arg2 reformulates Arg1.",
    "summary": "Arg2 summarises Arg1.",
    "organization": "textual organization (heading, preparation, phatic).",
    "preparation": "Arg1 prepares the reader for Arg2.",
    "topic": "topic management (question, solutionhood, shift).",
    "topic-comment": "Arg2 comments on the topic set in Arg1.",
    "topic-shift": "the topic shifts between the units.",
    "topic-drift": "the topic drifts between the units.",
    "evaluation": "Arg2 evaluates/assesses Arg1.",
    "interpretation": "Arg2 interprets Arg1.",
    "comment": "Arg2 comments on Arg1.",
    "condition": "Arg2 is a condition for Arg1 (if).",
    "contingency": "one unit is a condition for the other.",
    "hypothetical": "Arg2 states a hypothetical situation.",
    "otherwise": "Arg2 states what holds otherwise (else).",
    "unless": "Arg2 is a negative condition (unless).",
    "definition": "Arg2 defines a term in Arg1.",
    "example": "Arg2 gives an example of Arg1.",
    "solutionhood": "Arg2 is a solution to a problem/question in Arg1.",
    "problem-solution": "Arg2 is a solution to the problem in Arg1.",
    "question-answer": "Arg2 answers the question in Arg1.",
    "statement-response": "Arg2 responds to the statement in Arg1.",
    "conjunction": "the units both hold and are jointly relevant (and).",
    "disjunction": "the units are alternatives (or).",
    "conclusion": "Arg2 draws a conclusion from Arg1.",
    "textual-organization": "textual/structural organization of the document.",
    "textualorganization": "textual/structural organization of the document.",
    "temporal": "the units are related in time.",
    "temporal-before": "Arg1 happens before Arg2.",
    "temporal-after": "Arg1 happens after Arg2.",
    "temporal-same-time": "the units overlap in time.",
    "inverted-sequence": "the units are in reverse temporal sequence.",
    "nonvolitional-cause": "Arg2 is a non-deliberate cause of Arg1.",
    "nonvolitional-result": "Arg2 is a non-deliberate result of Arg1.",
    "volitional-cause": "Arg2 is a deliberate cause of Arg1.",
    "volitional-result": "Arg2 is a deliberate result of Arg1.",
    "parenthetical": "Arg2 is a parenthetical aside to Arg1.",
    "preference": "Arg1 is preferred over Arg2.",
}


def _rst(label):
    p = str(label).lower().rstrip("*")
    # strip RST-DT nucleus/satellite/embedded markers: -e, -n, -s, -N, -S, -M
    for suf in ("-e", "-n", "-s", "-m"):
        if p.endswith(suf):
            p = p[:-len(suf)]
    d = _RST.get(p)
    if not d:                                # try the class head before the first '-'
        d = _RST.get(p.split("-")[0])
    return d or ""


# ── SciDTB discourse-dependency (DEP) ───────────────────────────────────────
_DEP = {
    "attribution": "one unit attributes a statement to a source.",
    "background": "Arg2 gives background for Arg1.",
    "bg-general": "Arg2 gives general background for Arg1.",
    "bg-compare": "Arg2 gives comparative background for Arg1.",
    "bg-goal": "Arg2 gives the goal/motivation background for Arg1.",
    "cause": "Arg2 is the cause of Arg1.",
    "cause-result": "a cause–result link between the units.",
    "result": "Arg2 is the result of Arg1.",
    "comparison": "the units are compared.",
    "contrast": "the units contrast.",
    "condition": "Arg2 is a condition for Arg1 (if).",
    "elaboration": "Arg2 adds detail about Arg1.",
    "elab-addition": "Arg2 adds further information to Arg1.",
    "elab-aspect": "Arg2 elaborates an aspect of Arg1.",
    "elab-definition": "Arg2 defines a term in Arg1.",
    "elab-enumember": "Arg2 enumerates members of a set in Arg1.",
    "elab-example": "Arg2 gives an example of Arg1.",
    "elab-process_step": "Arg2 is a step in the process described in Arg1.",
    "enablement": "Arg2 enables the action in Arg1.",
    "evaluation": "Arg2 evaluates Arg1.",
    "exp-evidence": "Arg2 gives evidence for Arg1.",
    "exp-reason": "Arg2 gives the reason for Arg1.",
    "explanation": "Arg2 explains Arg1.",
    "findings": "Arg2 states the findings.",
    "joint": "the units are coordinated (and).",
    "manner-means": "Arg2 gives the manner/means of Arg1.",
    "progression": "the units form a progression.",
    "summary": "Arg2 summarises Arg1.",
    "temporal": "the units are related in time.",
    "textual-organization": "structural organization of the text.",
}


# ── ISO 24617-8 DR-core ─────────────────────────────────────────────────────
_ISO = {
    "asynchrony_before": "Arg1 precedes Arg2 in time.",
    "cause_result": "Arg2 is the result of Arg1.",
    "concession_expectation-raiser": "Arg1 raises an expectation that Arg2 denies (although).",
    "condition_consequent": "Arg2 is the consequent of the condition in Arg1 (if).",
    "conjunction": "the units both hold and are jointly relevant (and).",
    "contrast": "the units contrast.",
    "disjunction": "the units are alternatives (or).",
    "elaboration_broad": "Arg2 elaborates Arg1.",
    "exception_regular": "Arg2 is an exception to Arg1 (except).",
    "exemplification_set": "Arg2 exemplifies Arg1.",
    "expansion_narrative": "Arg2 continues the narrative from Arg1.",
    "manner_achievement": "Arg2 gives the manner of achieving Arg1.",
    "negativecondition_consequent": "Arg2 is the consequent of a negative condition (unless).",
    "purpose_enablement": "Arg2 is the purpose that Arg1 enables.",
    "restatement": "Arg2 restates Arg1.",
    "similarity": "Arg1 and Arg2 are similar.",
    "substitution_disfavoured-alternative": "Arg2 is chosen over a disfavoured alternative.",
    "synchrony": "Arg1 and Arg2 overlap in time.",
}


# ── SDRT extras (French / lowercase / variants beyond relations._REL_DEF) ────
_SDRT = {
    "attribution": "one unit attributes a statement/attitude to a source.",
    "frame": "Arg2 sets a frame (setting/topic) for Arg1.",
    "flashback": "Arg2 narrates an earlier event (reverse narration).",
    "goal": "Arg2 is the goal/purpose of Arg1.",
    "temploc": "Arg2 gives the temporal location of Arg1.",
    "e-elaboration": "Arg2 elaborates an entity mentioned in Arg1.",
}


def _iso(label):
    return _ISO.get(str(label).lower(), "")


def _dep(label):
    return _DEP.get(str(label).lower(), "")


def _sdrt(label):
    return _SDRT.get(str(label).lower(), "")


def define(label, framework=""):
    """One-line codebook gloss for a relation label in the given framework, or ""."""
    if not label:
        return ""
    fw = (framework or "").lower()
    if fw == "pdtb":
        return _pdtb(label)
    if fw in ("rst", "erst"):
        return _rst(label)
    if fw == "dep":
        return _dep(label)
    if fw == "iso":
        return _iso(label)
    if fw == "sdrt":
        return _sdrt(label)
    # unknown framework: try each
    for fn in (_pdtb, _rst, _dep, _iso, _sdrt):
        d = fn(label)
        if d:
            return d
    return ""
