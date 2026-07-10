"""
suggestion_engine.py
=============
Minimal Flask HTTP service exposing the DDPE explanation student (Liu et al.,
2025, "Enhancing Multi-party Dialogue Discourse Parsing with Explanation
Generation") so the DSDP/MuDiPA webapp can use it as the `ddpe` explanation engine
for STAC / Molweni.

The DDPE student = a LoRA adapter on LLaMA-3-8B. The adapters are vendored in
`ext/DDPE/{STAC,Molweni}/`; the base is the ungated mirror
`NousResearch/Meta-Llama-3-8B`.

⚠️ NOTE: the DDPE repo ships ONLY the adapters (no inference code / training
prompt template). The exact training format is not published, so the prompt
below is a faithful-intent RECONSTRUCTION (per-arc link + relation explanation
with [u_i]/[u_j] localization + contrastive), not the byte-exact training
format. Output uses the real DDPE-fine-tuned weights. The webapp falls back to
the zero-shot LLM if this server is unavailable or the dataset is not STAC/Molweni.

HARD CONSTRAINT: runs on GPU 1 ONLY (CUDA_VISIBLE_DEVICES=1, set before torch).

Endpoints
---------
GET  /health           -> {status, model_loaded, dataset, device, vram_allocated_mb}
POST /ddpe/load        -> {dataset:"stac"|"molweni"}  (loads base+adapter on GPU 1)
POST /ddpe/unload      -> frees VRAM
POST /ddpe/explain     -> {edus:[{speaker,text}], source, target} ->
                          {parent, link:{explanation}, relation:{candidate,explanation},
                           contrastive:{why_not,explanation}}

Usage
-----
    .venv-torch\\Scripts\\python.exe src\\suggestion_engine.py  [--port 8092]
"""

import os

# --- GPU 1 ONLY. Must run before importing torch. ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# (2) Reduce VRAM reservation/fragmentation from the caching allocator: expandable
# segments let CUDA give freed blocks back instead of holding a large reserved pool.
# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import re
import threading

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # servers/ -> project root (ext/, .env)


def _load_hf_token():
    """HF token for the gated OFFICIAL base — from env, else from .env (the repo's
    .env is a bare `hf_...` token on line 1, or KEY=value). Never logged."""
    for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(v):
            return os.environ[v]
    envf = os.path.join(_REPO_ROOT, ".env")
    if os.path.isfile(envf):
        for line in open(envf, encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split("=", 1)[1].strip().strip('"\'') if ("=" in s and "hf_" not in s.split("=")[0]) else s
            if tok.startswith("hf_"):
                os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
                return tok
    return None


HF_TOKEN = _load_hf_token()
# DDPE's LoRA was trained on the OFFICIAL meta-llama base → use it (the user now has
# gated access). Override with DSDP_LLAMA3_BASE if needed.
DEFAULT_BASE = os.environ.get("DSDP_LLAMA3_BASE", "meta-llama/Meta-Llama-3.1-8B")
ADAPTER_DIRS = {
    "stac":    os.path.join(_REPO_ROOT, "ext", "DDPE", "STAC"),
    "molweni": os.path.join(_REPO_ROOT, "ext", "DDPE", "Molweni"),
}

app = Flask(__name__)
_lock = threading.Lock()
_state = {"model": None, "tok": None, "dataset": None}


# (1) After every request, hand freed CUDA blocks back to the allocator so the reserved
# pool doesn't grow to the peak of a long teacher-forced scoring pass and sit there idle.
@app.after_request
def _free_cuda(resp):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return resp

# DDPE student (SPM) format — reconstructed verbatim from the paper (Liu et al.,
# COLING 2025, Table 6 / Table 10). The model is trained to COMPLETE, after a
# `label:` arc prefix, `'Relation': <explanation with [i]/[j] token markers>`.
# Arc notation is [child]–>[parent]; for our arc parent p -> child t the label is
# `[t]–>[p]`. We feed dialogue history D_<=t and let the LoRA fill relation+explanation.
PROMPT = (
    "Given the following dialogue and the relation list between utterances, provide "
    "an explanation for each relation label, and only output the following explanation "
    "content. Pay attention! Don't add anything else.\n"
    "Example output: [1]–>[0]:'Question-answer_pair': [1]Thomas answered [0]William's "
    "proposal with 'no'.\n"
    "dialogue:\n{dialogue}\n"
    "label:\n[{t}]–>[{p}]:")


def _vram_mb():
    try:
        return round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception:
        return 0.0


def _load(dataset):
    dataset = (dataset or "stac").lower()
    adir = ADAPTER_DIRS.get(dataset)
    if not adir or not os.path.isdir(adir):
        return False, f"no DDPE adapter for dataset '{dataset}'"
    if _state["model"] is not None and _state["dataset"] == dataset:
        return True, "already loaded"
    _unload()
    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        DEFAULT_BASE, torch_dtype=torch.bfloat16, device_map={"": 0}, token=HF_TOKEN)
    model = PeftModel.from_pretrained(base, adir)
    # Fold the LoRA into the base → no per-forward PEFT hooks → faster inference.
    model = model.merge_and_unload()
    model.eval()
    # Inference server: params never need grad. Disabling it lets /ddpe/ig run
    # Integrated Gradients w.r.t. inputs_embeds without storing 8B param grads.
    for prm in model.parameters():
        prm.requires_grad_(False)
    _state.update(model=model, tok=tok, dataset=dataset)
    return True, "loaded (merged)"


def _unload():
    if _state["model"] is not None:
        del _state["model"]
        _state["model"] = None
    _state["tok"] = None
    _state["dataset"] = None
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _state["model"] is not None,
                    "dataset": _state["dataset"],
                    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                    "vram_allocated_mb": _vram_mb()})


# Example DAGs to try on the fly (STAC-style short dialogues).
EXAMPLE_DAGS = [
    {"name": "Catan — ore Q/A", "dataset": "stac",
     "edus": [{"speaker": "Nancy", "text": "do u know how to trade with the port?"},
              {"speaker": "skinnylinny", "text": "Anyone got a spare ore?"},
              {"speaker": "Chameleon", "text": "Yes"}],
     "arcs": [{"source": 1, "target": 2}, {"source": 0, "target": 1}]},
    {"name": "Catan — clay request", "dataset": "stac",
     "edus": [{"speaker": "Gaeilgeoir", "text": "anyone have clay? :)"},
              {"speaker": "Gaeilgeoir", "text": "I have wheat and wood"},
              {"speaker": "yiin", "text": "no sorry"}],
     "arcs": [{"source": 0, "target": 2}, {"source": 0, "target": 1}]},
    {"name": "4-EDU thread", "dataset": "stac",
     "edus": [{"speaker": "A", "text": "anyone want to trade sheep?"},
              {"speaker": "B", "text": "what do you want for it?"},
              {"speaker": "A", "text": "wood or brick"},
              {"speaker": "B", "text": "ok I'll give you wood"}],
     "arcs": [{"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 3}]},
    {"name": "8-EDU multi-party (longer)", "dataset": "stac",
     "edus": [{"speaker": "Tomm", "text": "anyone got wheat to spare?"},
              {"speaker": "gotwood4sheep", "text": "i have some"},
              {"speaker": "Tomm", "text": "what do you want for 2 wheat?"},
              {"speaker": "gotwood4sheep", "text": "an ore"},
              {"speaker": "Tomm", "text": "i dont have ore"},
              {"speaker": "dmm", "text": "i can do ore for wheat"},
              {"speaker": "gotwood4sheep", "text": "ok deal with dmm then"},
              {"speaker": "Tomm", "text": "great, thanks both"}],
     "arcs": [{"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 2, "target": 3},
              {"source": 3, "target": 4}, {"source": 0, "target": 5}, {"source": 5, "target": 6},
              {"source": 6, "target": 7}]},
]


@app.route("/ddpe/examples")
def ddpe_examples():
    return jsonify(EXAMPLE_DAGS)


@app.route("/ddpe/embed", methods=["POST"])
def ddpe_embed():
    """Dense arc embedding in the PARSER's OWN latent space, taken at the
    RELATION-decision point of the specific arc (source p, target t): we build
    the prefix '...label:\\n[t]–>[p]:' — exactly what the scorer conditions on
    before emitting the relation string — and return the last-layer hidden state
    at its final token. This is the representation from which DDPE generates the
    (possibly wrong) relation, so it depends on BOTH p and t and lives where the
    gold/pred confusion arises. Two arcs close here are ones the parser labels
    alike → errs alike: the right space for retrieving relevant error examples.
    If `source` is omitted, falls back to the link-decision point '[t]–>'.
    L2-normalised, so downstream cosine = dot product.
    Input {items:[{edus, source, target}], dataset?} -> {embeddings, dim}."""
    payload = request.json or {}
    items = payload.get("items") or []
    dataset = (payload.get("dataset") or "stac").lower()
    if not items:
        return jsonify({"error": "items[] required"}), 400
    with _lock:
        if _state["model"] is None or _state["dataset"] != dataset:
            ok, msg = _load(dataset)
            if not ok:
                return jsonify({"error": msg}), 503
        model, tok = _state["model"], _state["tok"]
        header = PROMPT.split("dialogue:")[0]
        out = []
        with torch.no_grad():
            for it in items:
                edus = it["edus"]
                t = int(it["target"])
                p = it.get("source")
                spk = [e.get("speaker", "") for e in edus]
                txt = [e.get("text", "") for e in edus]
                dialogue = "\n".join(f"[{i}] {spk[i]}: {txt[i]}" for i in range(t + 1))
                arc = f"[{t}]–>[{int(p)}]:" if p is not None else f"[{t}]–>"
                prefix = f"{header}dialogue:\n{dialogue}\nlabel:\n{arc}"
                ids = torch.tensor([tok(prefix)["input_ids"]], device=model.device)
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
                v = hs[0, -1, :].float()
                v = v / (v.norm() + 1e-8)
                out.append(v.cpu().tolist())
    return jsonify({"embeddings": out, "dim": len(out[0]) if out else 0,
                    "engine": "ddpe", "dataset": dataset})


@app.route("/")
def test_page():
    return TEST_HTML


@app.route("/ddpe/load", methods=["POST"])
def ddpe_load():
    ds = (request.json or {}).get("dataset", "stac")
    with _lock:
        ok, msg = _load(ds)
    return jsonify({"ok": ok, "msg": msg, "dataset": _state["dataset"],
                    "vram_allocated_mb": _vram_mb()})


@app.route("/ddpe/unload", methods=["POST"])
def ddpe_unload():
    with _lock:
        _unload()
    return jsonify({"ok": True, "vram_allocated_mb": _vram_mb()})


@app.route("/ddpe/explain", methods=["POST"])
def ddpe_explain():
    payload = request.json or {}
    edus = payload["edus"]
    source = int(payload["source"])
    target = int(payload["target"])
    dataset = (payload.get("dataset") or "stac").lower()
    spk = [e.get("speaker", "") for e in edus]
    txt = [e.get("text", "") for e in edus]
    with _lock:
        if _state["model"] is None or _state["dataset"] != dataset:
            ok, msg = _load(dataset)
            if not ok:
                return jsonify({"error": msg}), 503
        model, tok = _state["model"], _state["tok"]
        # dialogue history D_<=t as "[i] Speaker: text" (DDPE format, Table 10)
        dialogue = "\n".join(f"[{i}] {spk[i]}: {txt[i]}" for i in range(target + 1))

        def _gen(prompt, n):
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=n, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Step 1 — predict the relation by completing the arc prefix `[t]–>[p]:`.
        g1 = _gen(PROMPT.format(dialogue=dialogue, t=target, p=source), 24)
        rel = re.sub(r"[\"':].*$", "", g1.strip().split("\n")[0]).strip().strip("'\" ")
        # Step 2 — given the full labeled arc, let the student produce the explanation.
        prompt2 = (PROMPT.format(dialogue=dialogue, t=target, p=source)
                   + f"'{rel}':")
        gen = _gen(prompt2, 200)
    expl = gen.strip().split("\n")[0].strip()[:600]
    return jsonify({
        "parent": source,
        # DDPE emits ONE explanation covering the (link+relation) arc; surface it on both.
        "link": {"explanation": expl, "confidence": 0.0, "score": None},
        "relation": {"candidate": rel, "explanation": expl, "confidence": 0.0, "fit": None},
        "contrastive": {"why_not": "", "explanation": ""},
        "engine": "ddpe", "raw": gen[:600],
    })


@app.route("/ddpe/parse", methods=["POST"])
def ddpe_parse():
    """INCREMENTAL parsing: DDPE is left-to-right — for the current utterance u_t it
    PREDICTS the parent + relation from the dialogue history D_<=t (it is NOT given
    the parent). The student completes the open arc prefix `[t]–>` → `[p]:'rel': …`.
    Input {edus:[{speaker,text}], target, dataset} → {target, parent, relation, raw}."""
    payload = request.json or {}
    edus = payload["edus"]
    target = int(payload["target"])
    dataset = (payload.get("dataset") or "stac").lower()
    spk = [e.get("speaker", "") for e in edus]
    txt = [e.get("text", "") for e in edus]
    with _lock:
        if _state["model"] is None or _state["dataset"] != dataset:
            ok, msg = _load(dataset)
            if not ok:
                return jsonify({"error": msg}), 503
        model, tok = _state["model"], _state["tok"]
        dialogue = "\n".join(f"[{i}] {spk[i]}: {txt[i]}" for i in range(target + 1))
        header = PROMPT.split("dialogue:")[0]                 # instruction + example
        prompt = f"{header}dialogue:\n{dialogue}\nlabel:\n[{target}]–>"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    line = gen.strip().split("\n")[0]
    pm = re.match(r"\s*\[(\d+)\]", line)
    parent = int(pm.group(1)) if pm else None
    rm = re.search(r"'([^']+)'", line)
    rel = (rm.group(1).strip() if rm
           else re.sub(r"^\s*\[\d+\]\s*:?\s*", "", line).split(":")[0].strip())
    expl = (line[rm.end():].lstrip(": ").strip() if rm else "")[:600]
    valid = parent is not None and 0 <= parent < target
    return jsonify({"target": target, "parent": parent, "valid": valid,
                    "relation": rel, "explanation": expl, "engine": "ddpe", "raw": gen[:400]})


@app.route("/ddpe/score", methods=["POST"])
def ddpe_score():
    """CONSTRAINED link scoring: only valid parents p < t are considered (no free
    generation, so no out-of-range predictions). For each candidate p we score the
    arc `[t]–>[p]:` by the model's average token log-prob (teacher-forced), then
    softmax-normalise → {links:[{source,score 0..1}]} sorted by score. Robust ranking
    for the 'suggest links' feature. Input {edus, target, dataset, window?}."""
    payload = request.json or {}
    edus = payload["edus"]
    target = int(payload["target"])
    dataset = (payload.get("dataset") or "stac").lower()
    window = int(payload.get("window", 10))
    spk = [e.get("speaker", "") for e in edus]
    txt = [e.get("text", "") for e in edus]
    with _lock:
        if _state["model"] is None or _state["dataset"] != dataset:
            ok, msg = _load(dataset)
            if not ok:
                return jsonify({"error": msg}), 503
        model, tok = _state["model"], _state["tok"]
        dialogue = "\n".join(f"[{i}] {spk[i]}: {txt[i]}" for i in range(target + 1))
        header = PROMPT.split("dialogue:")[0]
        prefix = f"{header}dialogue:\n{dialogue}\nlabel:\n[{target}]–>"
        pre_ids = tok(prefix)["input_ids"]
        plen = len(pre_ids)
        lo = max(0, target - window)
        cands = list(range(lo, target))
        conts = [tok(f"[{p}]:", add_special_tokens=False)["input_ids"] for p in cands]
        # ONE batched forward pass over [prefix + cont_p] for every candidate p.
        maxc = max((len(c) for c in conts), default=1)
        pad = tok.pad_token_id
        ids_b, mask_b = [], []
        for c in conts:
            ids_b.append(pre_ids + c + [pad] * (maxc - len(c)))
            mask_b.append([1] * (plen + len(c)) + [0] * (maxc - len(c)))
        ids = torch.tensor(ids_b, device=model.device)
        am = torch.tensor(mask_b, device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=am).logits      # [B, L, V]
        logp = torch.log_softmax(logits, dim=-1)
        rows = []
        for i, c in enumerate(conts):
            tot = sum(logp[i, plen + j - 1, ids[i, plen + j]].item() for j in range(len(c)))
            rows.append({"source": cands[i], "lp": tot / max(1, len(c))})
    mx = max((r["lp"] for r in rows), default=0.0)
    exps = [pow(2.718281828, r["lp"] - mx) for r in rows]
    Z = sum(exps) or 1.0
    for r, e in zip(rows, exps):
        r["score"] = round(e / Z, 3)
    rows.sort(key=lambda r: -r["score"])
    return jsonify({"target": target,
                    "links": [{"source": r["source"], "score": r["score"]} for r in rows],
                    "engine": "ddpe"})


# DDPE relation inventory (Liu et al. 2025, Table 9) — the strings the model emits.
DDPE_RELATIONS = ["Question-answer_pair", "Comment", "Acknowledgement", "Elaboration",
                  "Clarification_question", "Continuation", "Result", "Explanation",
                  "Correction", "Contrast", "Parallel", "Conditional", "Alternation",
                  "Background", "Narration", "Q-Elab"]


@app.route("/ddpe/score_rel", methods=["POST"])
def ddpe_score_rel():
    """Relation prediction SCORE for a given arc p→t (NO explanation). Scores each
    candidate SDRT relation r by the model's avg token log-prob of `[t]–>[p]:'r'`,
    softmax-normalised → {relations:[{relation,score 0..1}]} sorted. One batched
    forward over the 16 relations. Input {edus, source, target, dataset, relations?}."""
    payload = request.json or {}
    edus = payload["edus"]
    source = int(payload["source"])
    target = int(payload["target"])
    dataset = (payload.get("dataset") or "stac").lower()
    rels = payload.get("relations") or DDPE_RELATIONS
    spk = [e.get("speaker", "") for e in edus]
    txt = [e.get("text", "") for e in edus]
    with _lock:
        if _state["model"] is None or _state["dataset"] != dataset:
            ok, msg = _load(dataset)
            if not ok:
                return jsonify({"error": msg}), 503
        model, tok = _state["model"], _state["tok"]
        dialogue = "\n".join(f"[{i}] {spk[i]}: {txt[i]}" for i in range(target + 1))
        header = PROMPT.split("dialogue:")[0]
        prefix = f"{header}dialogue:\n{dialogue}\nlabel:\n[{target}]–>[{source}]:"
        pre_ids = tok(prefix)["input_ids"]
        plen = len(pre_ids)
        conts = [tok(f"'{r}'", add_special_tokens=False)["input_ids"] for r in rels]
        maxc = max((len(c) for c in conts), default=1)
        pad = tok.pad_token_id
        ids_b, mask_b = [], []
        for c in conts:
            ids_b.append(pre_ids + c + [pad] * (maxc - len(c)))
            mask_b.append([1] * (plen + len(c)) + [0] * (maxc - len(c)))
        ids = torch.tensor(ids_b, device=model.device)
        am = torch.tensor(mask_b, device=model.device)
        with torch.no_grad():
            logp = torch.log_softmax(model(input_ids=ids, attention_mask=am).logits, dim=-1)
        lps = [sum(logp[i, plen + j - 1, ids[i, plen + j]].item() for j in range(len(c))) / max(1, len(c))
               for i, c in enumerate(conts)]
    mx = max(lps, default=0.0)
    exps = [pow(2.718281828, lp - mx) for lp in lps]
    Z = sum(exps) or 1.0
    out = sorted([{"relation": rels[i], "score": round(exps[i] / Z, 3)} for i in range(len(rels))],
                 key=lambda r: -r["score"])
    return jsonify({"source": source, "target": target, "relations": out, "engine": "ddpe"})


# canonical (confusion-cell) name -> the surface string DDPE actually emits.
# Only three differ; the rest are identical in both inventories.
_CANON2SURFACE = {
    "QAP": "Question-answer_pair",
    "Clarification_Q": "Clarification_question",
    "Q-Elab": "Q-Elab",
}


def _rel_surface(rel):
    """canonical relation name -> the surface string DDPE actually emits."""
    return _CANON2SURFACE.get(rel, rel)


@app.route("/ddpe/ig", methods=["POST"])
def ddpe_ig():
    """Integrated Gradients attribution for a parser MISLABELLING. Target scalar
    is Δ = logp(pred) − logp(gold): what pushed the parser toward the WRONG
    relation over the correct one for arc (source p, target t). We integrate the
    gradient of Δ w.r.t. the prefix input embeddings from a zero baseline, sum
    over the hidden dim → a signed attribution per input token, then aggregate to
    EDUs via char offsets. Positive = drove the error. Returns normalised
    contributions (|attr| as % of total) per EDU and the top tokens.
    Input {edus, source, target, pred, gold, dataset?, steps?}."""
    payload = request.json or {}
    edus = payload["edus"]
    p = int(payload["source"]); t = int(payload["target"])
    pred = payload["pred"]; gold = payload["gold"]
    dataset = (payload.get("dataset") or "stac").lower()
    steps = int(payload.get("steps", 16))
    spk = [e.get("speaker", "") for e in edus]
    txt = [e.get("text", "") for e in edus]
    with _lock:
        if _state["model"] is None or _state["dataset"] != dataset:
            ok, msg = _load(dataset)
            if not ok:
                return jsonify({"error": msg}), 503
        model, tok = _state["model"], _state["tok"]
        emb_layer = model.get_input_embeddings()
        header = PROMPT.split("dialogue:")[0]
        # build prefix, tracking char spans per EDU: the full line span (for
        # per-EDU aggregation) and the text-only span (for content top_tokens).
        lines, spans, tspans, cur = [], [], [], len(f"{header}dialogue:\n")
        for i in range(t + 1):
            head_i = f"[{i}] {spk[i]}: "
            ln = head_i + txt[i]
            spans.append((i, cur, cur + len(ln)))                    # whole line
            tspans.append((i, cur + len(head_i), cur + len(ln)))     # utterance text only
            lines.append(ln); cur += len(ln) + 1        # +1 for the newline
        prefix = f"{header}dialogue:\n" + "\n".join(lines) + f"\nlabel:\n[{t}]–>[{p}]:"
        try:
            enc = tok(prefix, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
        except Exception:                 # slow tokenizer: no offsets → tokens only
            enc = tok(prefix); offsets = None
        pre_ids = enc["input_ids"]
        plen = len(pre_ids)
        pre_ids_t = torch.tensor([pre_ids], device=model.device)

        def cont_ids(rel):
            return tok(f"'{_rel_surface(rel)}'", add_special_tokens=False)["input_ids"]
        pc, gc = cont_ids(pred), cont_ids(gold)
        pc_t = torch.tensor([pc], device=model.device)
        gc_t = torch.tensor([gc], device=model.device)

        emb_pre = emb_layer(pre_ids_t).detach()          # [1, L, H]
        emb_pc  = emb_layer(pc_t).detach()
        emb_gc  = emb_layer(gc_t).detach()
        baseline = torch.zeros_like(emb_pre)

        def cont_logp(x, cont_emb, cont_id_list):
            # use_cache=False: no KV cache retained in the grad graph.
            # Only log_softmax the rows that predict the continuation tokens
            # (len_cont x V), never the full [L x V] float32 logits.
            ids_full = torch.cat([x, cont_emb], dim=1)
            logits = model(inputs_embeds=ids_full, use_cache=False).logits
            s = 0.0
            for j, cid in enumerate(cont_id_list):
                row = logits[0, plen + j - 1].float()          # [V]
                s = s + torch.log_softmax(row, dim=-1)[cid]
            return s / max(1, len(cont_id_list))

        total = torch.zeros_like(emb_pre)
        for k in range(1, steps + 1):
            alpha = k / steps
            x = (baseline + alpha * (emb_pre - baseline)).detach().requires_grad_(True)
            delta = cont_logp(x, emb_pc, pc) - cont_logp(x, emb_gc, gc)
            g, = torch.autograd.grad(delta, x)
            total = total + g.detach()
            del x, delta, g
        avg_grad = total / steps
        attr = ((emb_pre - baseline) * avg_grad).sum(-1)[0].float().cpu().tolist()  # [L]
        # release the per-request graph tensors before returning
        del total, avg_grad, emb_pre, emb_pc, emb_gc, baseline
        torch.cuda.empty_cache()

    # aggregate token attributions to EDUs via char offsets (if available)
    def _clean(tid):
        # decode() merges byte-level pieces correctly (LLaMA BPE shows Ġ/Ċ raw)
        return tok.decode([tid]).replace("\n", "\\n").strip()
    edu_attr = {i: 0.0 for i in range(t + 1)}
    toks = []   # ONLY tokens inside an EDU's utterance text (content words),
                # tagged with their EDU index — scaffolding/markers excluded.
    if offsets is not None:
        for idx, (a, b) in enumerate(offsets):
            if a == b:      # special/no-span token
                continue
            val = attr[idx]
            for i, s0, s1 in spans:                 # per-EDU total = whole line
                if a >= s0 and b <= s1:
                    edu_attr[i] += val
                    break
            for i, s0, s1 in tspans:                # top_tokens = text-only spans
                if a >= s0 and b <= s1:
                    tk = _clean(pre_ids[idx])
                    if tk:
                        toks.append({"tok": tk, "attr": round(val, 4), "edu": i})
                    break
    denom = sum(abs(v) for v in edu_attr.values()) or 1.0
    edus_out = sorted(
        [{"edu": i, "speaker": spk[i], "text": txt[i],
          "attr": round(v, 4), "pct": round(100 * v / denom, 1)}
         for i, v in edu_attr.items()], key=lambda d: -abs(d["attr"]))
    top_tokens = sorted(toks, key=lambda d: -abs(d["attr"]))[:12]
    return jsonify({"source": p, "target": t, "pred": pred, "gold": gold,
                    "steps": steps, "edus": edus_out, "top_tokens": top_tokens,
                    "engine": "ddpe-ig"})


TEST_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>DDPE test</title><style>
 body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#0e1116;color:#e6e6e6;font-size:16px}
 header{padding:12px 18px;background:#161b22;border-bottom:1px solid #2a2f37;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 h1{font-size:16px;margin:0}
 main{padding:18px;max-width:1000px;margin:0 auto}
 textarea{width:100%;min-height:130px;background:#0b0e13;color:#e6e6e6;border:1px solid #2a2f37;border-radius:8px;padding:10px;font-family:ui-monospace,monospace;font-size:14px}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
 button,select,input{font-size:15px;padding:7px 12px;border-radius:7px;border:1px solid #2a2f37;background:#1c2230;color:#e6e6e6;cursor:pointer}
 button.primary{background:#2563eb;border-color:#2563eb}
 label{color:#9aa4b2;font-size:13px}
 #out{white-space:pre-wrap;background:#0b0e13;border:1px solid #2a2f37;border-radius:8px;padding:12px;margin-top:12px;min-height:60px}
 .pill{padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700}
 .ok{background:#0a2a1a;color:#34d399}.bad{background:#2a0f0f;color:#f87171}
 .rel{display:inline-block;background:#1e3a5f;color:#7dd3fc;padding:3px 9px;border-radius:6px;font-weight:700}
</style></head><body>
<header><h1>🧪 DDPE parser — live test</h1>
 <span>GPU 1 · LLaMA-3-8B + LoRA</span><span id="status" class="pill"></span>
 <span style="flex:1"></span>
 <label>dataset</label><select id="ds"><option>stac</option><option>molweni</option></select>
 <button id="load">Load DDPE</button><button id="unload">Unload</button></header>
<main>
 <div class="row"><label>Example:</label><select id="ex"></select><button id="useEx">Load example</button></div>
 <label>EDUs (one per line — <code>speaker: text</code>)</label>
 <textarea id="edus"></textarea>
 <p style="color:#9aa4b2;font-size:13px;margin:6px 0">DDPE parses <b>incrementally</b> (left-to-right): for each utterance <code>u_t</code> it <b>predicts</b> the parent + relation from the history <code>u_0..u_t</code> (it is NOT given the parent).</p>
 <div class="row">
   <label>target t</label><input id="tgt" type="number" value="2" style="width:70px">
   <button id="parse" class="primary">Parse u_t →</button>
   <button id="parseAll">Parse all (incremental)</button>
 </div>
 <div class="row">
   <button id="scoreLinks" class="primary">Score links → t</button>
   <button id="scoreRel">Score relations (src→t)</button>
   <span style="flex:1"></span>
   <label>arc src</label><input id="src" type="number" value="0" style="width:60px">
   <button id="explain">Explain arc</button>
 </div>
 <div id="out">Load the model, pick/edit a DAG, then <b>Parse</b>: DDPE predicts each EDU's parent.</div>
</main>
<script>
const $=s=>document.querySelector(s);
const setStatus=(t,ok)=>{const e=$("#status");e.textContent=t;e.className="pill "+(ok?"ok":"bad");};
async function refresh(){try{const h=await(await fetch("/health")).json();setStatus(h.model_loaded?("loaded: "+h.dataset+" ("+Math.round(h.vram_allocated_mb)+" MB)"):"not loaded",h.model_loaded);}catch(e){setStatus("server down",false);}}
function parseEdus(){return $("#edus").value.split("\\n").filter(l=>l.trim()).map(l=>{const i=l.indexOf(":");return i<0?{speaker:"?",text:l.trim()}:{speaker:l.slice(0,i).trim(),text:l.slice(i+1).trim()};});}
let EX=[];
fetch("/ddpe/examples").then(r=>r.json()).then(d=>{EX=d;$("#ex").innerHTML=d.map((e,i)=>`<option value="${i}">${e.name}</option>`).join("");useExample();});
function useExample(){const e=EX[$("#ex").value];if(!e)return;$("#edus").value=e.edus.map(u=>u.speaker+": "+u.text).join("\\n");$("#ds").value=e.dataset;if(e.arcs&&e.arcs[0]){$("#src").value=e.arcs[0].source;$("#tgt").value=e.arcs[0].target;}}
$("#useEx").onclick=useExample;
$("#load").onclick=async()=>{setStatus("loading…",true);const r=await(await fetch("/ddpe/load",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dataset:$("#ds").value})})).json();await refresh();};
$("#unload").onclick=async()=>{await fetch("/ddpe/unload",{method:"POST"});await refresh();};
async function parseOne(edus,t,ds){return await(await fetch("/ddpe/parse",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({edus,target:t,dataset:ds})})).json();}
$("#parse").onclick=async()=>{
  $("#out").textContent="parsing…";const edus=parseEdus(),ds=$("#ds").value,t=+$("#tgt").value;
  try{const t0=Date.now();const r=await parseOne(edus,t,ds);
    $("#out").innerHTML=`<b>parse u_${t}</b> (DDPE predicts the parent)  ·  ${((Date.now()-t0)/1000).toFixed(1)}s\\n\\n`+
      `predicted arc: <span class="rel">[${r.parent}] → [${t}]</span>  ${r.valid?"":"⚠ (out of range)"}\\nrelation: <span class="rel">${r.relation||"?"}</span>\\n\\nraw model output:\\n${r.raw||""}`;
  }catch(e){$("#out").textContent="ERROR: "+e.message+"\\n(model loaded? click Load DDPE)";}
};
$("#parseAll").onclick=async()=>{
  const edus=parseEdus(),ds=$("#ds").value,n=edus.length;let lines=[];const t0=Date.now();
  for(let t=1;t<n;t++){$("#out").textContent="parsing u_"+t+" / "+(n-1)+"…";try{const r=await parseOne(edus,t,ds);
    lines.push(`[${r.parent}] → [${t}]  :  ${r.relation||"?"}${r.valid?"":"  ⚠"}`);}catch(e){lines.push(`[?] → [${t}]  ERROR`);}}
  $("#out").innerHTML=`<b>incremental parse</b> (parent predicted per utterance)  ·  ${((Date.now()-t0)/1000).toFixed(1)}s\\n\\n`+lines.join("\\n");
};
function bar(s){const w=Math.round(s*100);return `<span style="display:inline-block;height:10px;width:${Math.max(2,w*1.4)}px;background:#38bdf8;border-radius:3px;vertical-align:middle"></span> ${(s*100).toFixed(1)}%`;}
$("#scoreLinks").onclick=async()=>{
  $("#out").textContent="scoring links…";const edus=parseEdus(),ds=$("#ds").value,t=+$("#tgt").value;
  try{const t0=Date.now();const r=await(await fetch("/ddpe/score",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({edus,target:t,dataset:ds})})).json();
    if(r.error){$("#out").textContent="ERROR: "+r.error;return;}
    $("#out").innerHTML=`<b>link scores → [${t}]</b> (constrained to parents &lt; ${t})  ·  ${((Date.now()-t0)/1000).toFixed(1)}s\\n\\n`+
      r.links.map(l=>`[${l.source}] → [${t}]   ${bar(l.score)}`).join("\\n");
  }catch(e){$("#out").textContent="ERROR: "+e.message+"\\n(model loaded? click Load DDPE)";}
};
$("#scoreRel").onclick=async()=>{
  $("#out").textContent="scoring relations…";const edus=parseEdus(),ds=$("#ds").value,s=+$("#src").value,t=+$("#tgt").value;
  try{const t0=Date.now();const r=await(await fetch("/ddpe/score_rel",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({edus,source:s,target:t,dataset:ds})})).json();
    if(r.error){$("#out").textContent="ERROR: "+r.error;return;}
    $("#out").innerHTML=`<b>relation scores for arc [${s}] → [${t}]</b>  ·  ${((Date.now()-t0)/1000).toFixed(1)}s\\n\\n`+
      r.relations.map(x=>`${x.relation.padEnd(24," ")} ${bar(x.score)}`).join("\\n");
  }catch(e){$("#out").textContent="ERROR: "+e.message;}
};
$("#explain").onclick=async()=>{
  $("#out").textContent="running…";const body={edus:parseEdus(),source:+$("#src").value,target:+$("#tgt").value,dataset:$("#ds").value};
  try{const t0=Date.now();const r=await(await fetch("/ddpe/explain",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
    $("#out").innerHTML=`<b>explain arc (${body.source} → ${body.target})</b>  ·  ${((Date.now()-t0)/1000).toFixed(1)}s\\n\\nrelation: <span class="rel">${r.relation?.candidate||"?"}</span>\\n\\nexplanation:\\n${r.relation?.explanation||"(empty)"}\\n\\nraw:\\n${r.raw||""}`;
  }catch(e){$("#out").textContent="ERROR: "+e.message;}
};
refresh();setInterval(refresh,5000);
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"[suggestion_engine] base={DEFAULT_BASE} adapters={list(ADAPTER_DIRS)} "
          f"on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} :{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
