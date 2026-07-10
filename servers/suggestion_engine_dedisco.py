"""
DeDisCo relation-classification server for MuDiPA.

Uses the Georgetown Qwen3-4B model fine-tuned on DISRPT 2025.
The model outputs 17 unified DISRPT labels; these are mapped back to
the 16 STAC-native SDRT labels and the probability is redistributed.

Endpoint: POST /dedisco/score_rel
  Input:  {"edus": [{"speaker":str,"text":str}, ...], "source": int, "target": int}
  Output: {"source":int, "target":int, "relations":[{"relation":str,"score":float}, ...]}

Usage:
    CUDA_VISIBLE_DEVICES=1 python suggestion_engine_dedisco.py [--port 8093] [--model MODEL_ID]
"""

import argparse
import os

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "JuNymphea/Georgetown-qwen3-4B-finetuned-for-disrpt2025"

# The 17 unified DISRPT 2025 relation labels (from decoder_w_aug.py)
DISRPT_LABELS = [
    "contrast", "condition", "mode", "organization", "frame", "temporal",
    "concession", "reformulation", "comment", "query", "attribution",
    "alternation", "purpose", "explanation", "elaboration", "causal",
    "conjunction",
]

# STAC-native SDRT labels (16)
STAC_RELATIONS = [
    "Acknowledgement", "Alternation", "Background", "Clarification_question",
    "Comment", "Conditional", "Continuation", "Contrast", "Correction",
    "Elaboration", "Explanation", "Narration", "Parallel", "Q-Elab",
    "QAP", "Result",
]

# DISRPT → STAC weighted mapping.
# Each entry maps a DISRPT label to {stac_label: weight} dicts.
# Weights reflect approximate STAC corpus frequencies within the group
# so that common labels (QAP, Continuation) aren't over-diluted by rare ones.
DISRPT_TO_STAC = {
    "contrast":      {"Contrast": 1.0},
    "condition":     {"Conditional": 1.0},
    "elaboration":   {"Elaboration": 1.0},
    "explanation":   {"Explanation": 1.0},
    "alternation":   {"Alternation": 1.0},
    "conjunction":   {"Continuation": 3.0, "Parallel": 1.0, "Narration": 1.0},
    "causal":        {"Result": 1.5, "Explanation": 1.0},
    "query":         {"QAP": 3.0, "Clarification_question": 1.0, "Q-Elab": 1.5},
    "comment":       {"Comment": 2.0, "Acknowledgement": 1.5},
    "reformulation": {"Correction": 1.5, "Elaboration": 1.0},
    "temporal":      {"Narration": 1.0},
    "concession":    {"Contrast": 1.0},
    "frame":         {"Background": 1.0},
    "mode":          {"Background": 1.0},
    "attribution":   {"Background": 1.0, "Comment": 1.0},
    "purpose":       {"Explanation": 1.5, "Result": 1.0},
    "organization":  {"Continuation": 1.0},
}

# ── Globals ────────────────────────────────────────────────────────────────────

_model     = None
_tokenizer = None
_device    = None

# ── Prompt builder (matches training format from decoder_w_aug.py) ─────────────

_SYSTEM = (
    "You are an expert in discourse relation classification. "
    "Given two discourse units and context, classify their relation. "
    "Output only the relation label from the provided list."
)

_LABELS_STR = ", ".join(DISRPT_LABELS)

def _build_prompt(edus, src, tgt):
    same_spk = "same" if edus[src]["speaker"] == edus[tgt]["speaker"] else "different"
    dist = tgt - src

    # Context window: 2 turns before src, 2 after tgt
    ctx_lines = []
    for i in range(max(0, src - 2), min(len(edus), tgt + 3)):
        ctx_lines.append(f"[{i}] {edus[i]['speaker']}: {edus[i]['text']}")
    context = "\n".join(ctx_lines)

    user = (
        f"## Language:\neng\n\n"
        f"## Corpus:\nstac\n\n"
        f"## Framework:\nsdrt\n\n"
        f"## Speaker:\n{same_spk} speaker\n\n"
        f"## Distance:\n{dist} unit(s)\n\n"
        f"## Context:\n{context}\n\n"
        f"## Labels:\n{_LABELS_STR}\n\n"
        f"## Direction:\n1>2\n\n"
        f"## Unit 1 (source):\n{edus[src]['speaker']}: {edus[src]['text']}\n\n"
        f"## Unit 2 (target):\n{edus[tgt]['speaker']}: {edus[tgt]['text']}\n\n"
        f"## Relation:"
    )
    return _SYSTEM, user


# ── Log-prob scoring with length normalisation ────────────────────────────────

def _score_disrpt(edus, src, tgt):
    """Return {disrpt_label: prob} for all 17 DISRPT labels."""
    sys_p, user_p = _build_prompt(edus, src, tgt)
    messages = [{"role": "system", "content": sys_p},
                {"role": "user",   "content": user_p}]
    prompt_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prompt_ids = _tokenizer.encode(prompt_text, add_special_tokens=False)

    # Build one sequence per DISRPT label: prompt + label tokens
    sequences  = []
    label_lens = []
    for rel in DISRPT_LABELS:
        lids = _tokenizer.encode(rel, add_special_tokens=False)
        sequences.append(prompt_ids + lids)
        label_lens.append(len(lids))

    # Left-pad to uniform length
    max_len = max(len(s) for s in sequences)
    pad_id  = _tokenizer.pad_token_id or _tokenizer.eos_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(s)) + s for s in sequences],
        dtype=torch.long, device=_device,
    )
    attn_mask = (input_ids != pad_id).long()

    with torch.no_grad():
        logits = _model(input_ids=input_ids, attention_mask=attn_mask).logits

    # Mean log-prob per label (length-normalised to avoid bias toward short labels)
    log_scores = []
    for i, ll in enumerate(label_lens):
        label_start = max_len - ll
        lp = 0.0
        for j in range(ll):
            pos      = label_start + j
            pred_pos = pos - 1
            tok_id   = input_ids[i, pos].item()
            lp += F.log_softmax(logits[i, pred_pos], dim=-1)[tok_id].item()
        log_scores.append(lp / ll)

    probs = F.softmax(torch.tensor(log_scores, dtype=torch.float32), dim=0).tolist()
    return dict(zip(DISRPT_LABELS, probs))


def _map_to_stac(disrpt_probs):
    """Redistribute DISRPT probabilities onto the 16 STAC labels using
    frequency-weighted mapping so common STAC labels aren't over-diluted."""
    stac_scores = {r: 0.0 for r in STAC_RELATIONS}
    for dlabel, prob in disrpt_probs.items():
        group = DISRPT_TO_STAC.get(dlabel)
        if not group:
            continue
        total_w = sum(group.values())
        for stac_label, w in group.items():
            if stac_label in stac_scores:
                stac_scores[stac_label] += prob * (w / total_w)

    total = sum(stac_scores.values()) or 1.0
    return sorted(
        [{"relation": r, "score": round(v / total, 4)} for r, v in stac_scores.items()],
        key=lambda x: -x["score"],
    )


# ── Flask server ───────────────────────────────────────────────────────────────

app = Flask(__name__)


def load_model(model_id, device):
    global _model, _tokenizer, _device
    print(f"Loading tokenizer from {model_id} ...", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    print(f"Loading model on {device} (bfloat16) ...", flush=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device)
    _model.eval()
    _device = device
    print("DeDisCo model ready.", flush=True)


@app.route("/dedisco/score_rel", methods=["POST"])
def score_rel():
    payload = request.json or {}
    edus = payload.get("edus", [])
    src  = int(payload.get("source", 0))
    tgt  = int(payload.get("target", 1))

    if not edus or src >= tgt or tgt >= len(edus):
        return jsonify({"error": "invalid source/target"}), 400

    disrpt_probs = _score_disrpt(edus, src, tgt)
    relations    = _map_to_stac(disrpt_probs)
    return jsonify({
        "source": src, "target": tgt,
        "relations": relations,
        "disrpt": sorted(disrpt_probs.items(), key=lambda x: -x[1])[:5],  # top-5 raw
    })


@app.route("/dedisco/health")
def health():
    return jsonify({"status": "ok", "model": DEFAULT_MODEL})


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port",  type=int, default=8093)
    p.add_argument("--model", default=os.environ.get("DEDISCO_MODEL", DEFAULT_MODEL))
    p.add_argument("--gpu",   default="cuda:0",
                   help="PyTorch device. Set CUDA_VISIBLE_DEVICES=1 externally.")
    args = p.parse_args()

    load_model(args.model, args.gpu)
    app.run(host="0.0.0.0", port=args.port, threaded=False)
