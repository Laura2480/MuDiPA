"""
Build a confusion-distribution example index from DDPE predictions on STAC.

The idea: instead of a hand-thresholded list of confusable relation pairs,
run the parser on a held-out in-distribution split (dev), compute the FULL
relation confusion matrix on correctly-linked arcs, and harvest one real
example instance per error. Each (gold, pred) cell then gets an example
quota PROPORTIONAL to its mass in the error distribution, so downstream
LLM reasoning generation can draw more (and more varied) examples for the
confusions the parser actually makes most often.

Output: data/stac_confusion_index.json
  {
    "meta":   {split, model_url, n_dialogues, n_correct_links, n_rel_errors, budget},
    "matrix": {gold_rel: {pred_rel: count}},          # full, diagonal included
    "cells":  [ {gold, pred, count, share, quota,
                 examples: [{dlg_id, src, tgt, gold, pred, distance,
                             same_speaker, src_speaker, src_text,
                             tgt_speaker, tgt_text, context}]} ]   # errors only
  }

Consumers import `load_index()` / `examples_for()` from this module
(see build_explanation_hints.py).

Usage (DDPE server must be running on :8092 with the stac adapter):
    python build_confusion_index.py                 # parse dev + build index
    python build_confusion_index.py --from-dump     # reuse saved predictions
    python build_confusion_index.py --budget 200    # total example allocation
"""
import argparse
import collections
import json
import time
from pathlib import Path

import requests

import relations
from eval_molweni_ddpe import lenient_parse, norm_rel

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
SUGGESTION_ENGINE_URL  = "http://127.0.0.1:8092/ddpe/parse"
INDEX_FILE = DATA_DIR / "stac_confusion_index.json"


# ── STAC data ─────────────────────────────────────────────────────────────────

def load_split(split):
    path = DATA_DIR / "stac_subindex" / f"{split}_subindex.json"
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ── DDPE parsing ─────────────────────────────────────────────────────────────

def parse_split(data, split):
    """Run DDPE incrementally over every dialogue; return per-dialogue preds."""
    per_dlg = []
    t0 = time.time()
    for di, dlg in enumerate(data):
        edus = [{"speaker": e["speaker"], "text": e["text"]} for e in dlg["edus"]]
        preds = []
        for t in range(1, len(edus)):
            try:
                resp = requests.post(
                    SUGGESTION_ENGINE_URL, json={"dataset": "stac", "target": t, "edus": edus},
                    timeout=300).json()
            except Exception:
                continue
            parent, rel = lenient_parse(resp.get("raw", ""))
            if parent is None or not (0 <= parent < t):
                continue
            preds.append([parent, t, rel])
        per_dlg.append({"id": dlg.get("id"), "n_edu": len(edus), "pred": preds})
        if (di + 1) % 10 == 0 or di + 1 == len(data):
            print(f"  {di+1}/{len(data)} dialogues  ({time.time()-t0:.0f}s)", flush=True)
    return per_dlg


# ── Confusion matrix + example harvest ───────────────────────────────────────

def context_lines(edus, src, tgt, pad=1, max_lines=8):
    """Dialogue excerpt around the arc, middle elided if the span is long."""
    lo, hi = max(0, src - pad), min(len(edus) - 1, tgt + pad)
    idxs = list(range(lo, hi + 1))
    if len(idxs) > max_lines:
        head = idxs[: max_lines // 2]
        tail = idxs[-(max_lines - max_lines // 2):]
        idxs = head + [None] + tail
    out = []
    for i in idxs:
        out.append("  ..." if i is None
                   else f"[{i}] {edus[i]['speaker']}: {edus[i]['text']}")
    return out


def harvest(data, per_dlg):
    """Full confusion matrix on correct links + one example per relation error."""
    by_id  = {d.get("id"): d for d in data}
    matrix = collections.defaultdict(collections.Counter)
    cell_examples = collections.defaultdict(list)
    n_correct_links = 0

    for row in per_dlg:
        dlg = by_id.get(row["id"])
        if dlg is None:
            continue
        edus = dlg["edus"]
        g_link = {(int(r["x"]), int(r["y"])): relations.normalize_relation(r["type"])
                  for r in dlg["relations"]}
        for p, c, pr in row["pred"]:
            if (p, c) not in g_link:
                continue                      # link error: not a relation confusion
            n_correct_links += 1
            gr, pr = g_link[(p, c)], norm_rel(pr)
            matrix[gr][pr] += 1
            if gr != pr:
                cell_examples[(gr, pr)].append({
                    "dlg_id": row["id"], "src": p, "tgt": c,
                    "gold": gr, "pred": pr,
                    "distance": c - p,
                    "same_speaker": edus[p]["speaker"] == edus[c]["speaker"],
                    "src_speaker": edus[p]["speaker"], "src_text": edus[p]["text"],
                    "tgt_speaker": edus[c]["speaker"], "tgt_text": edus[c]["text"],
                    "context": context_lines(edus, p, c),
                })
    return matrix, cell_examples, n_correct_links


def diversify(examples):
    """Order a cell's examples so consecutive picks vary: round-robin over
    dialogues, alternating short/long attachment distance within each."""
    by_dlg = collections.defaultdict(list)
    for ex in examples:
        by_dlg[ex["dlg_id"]].append(ex)
    for exs in by_dlg.values():
        exs.sort(key=lambda e: e["distance"])
        half = len(exs) // 2
        near, far = exs[:half], exs[half:][::-1]
        exs[:] = [x for pair in zip(far, near) for x in pair] + \
                 (far[len(near):] if len(far) > len(near) else near[len(far):])
    queues = sorted(by_dlg.values(), key=len, reverse=True)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def build_index(matrix, cell_examples, n_correct_links, budget, explain_budget,
                split, min_count=1):
    n_errors = sum(len(v) for v in cell_examples.values())
    kept = {k: v for k, v in cell_examples.items() if len(v) >= min_count}
    n_kept = sum(len(v) for v in kept.values())
    # shares are computed over the KEPT mass, so the budgets are spent on
    # reliable cells instead of being diluted by the singleton tail
    cells = []
    for (gr, pr), exs in sorted(kept.items(), key=lambda kv: -len(kv[1])):
        count = len(exs)
        share = count / n_kept if n_kept else 0.0
        quota = min(count, max(1, round(budget * share)))
        explain_quota = min(count, max(1, round(explain_budget * share)))
        cells.append({
            "gold": gr, "pred": pr, "count": count,
            "share": round(share, 4), "quota": quota,
            "explain_quota": explain_quota,
            "examples": diversify(exs),
        })
    return {
        "meta": {
            "split": split, "model_url": SUGGESTION_ENGINE_URL,
            "n_correct_links": n_correct_links,
            "n_rel_errors": n_errors, "budget": budget,
            "explain_budget": explain_budget,
            "min_count": min_count,
            "n_cells_total": len(cell_examples), "n_cells_kept": len(kept),
            "n_errors_kept": n_kept,
            "tail_dropped": {"cells": len(cell_examples) - len(kept),
                             "errors": n_errors - n_kept},
        },
        "matrix": {g: dict(c) for g, c in sorted(matrix.items())},
        "cells": cells,
    }


# ── Per-cell statistics / cost report ────────────────────────────────────────

# Claude generation cost model for one explained example (Sonnet 5, Batch API,
# intro pricing through 2026-08-31): measured on build_explanation_hints.py.
TOK_IN, TOK_OUT = 850, 300
PRICE_IN, PRICE_OUT = 1.0, 5.0        # $/MTok, batch

def cell_report(index):
    cost_one = (TOK_IN * PRICE_IN + TOK_OUT * PRICE_OUT) / 1e6
    lines = [f"{'gold -> pred':36s} {'n':>4s} {'share':>6s} {'idx':>4s} "
             f"{'expl':>5s} {'est.$':>7s}"]
    tot_q = tot_e = 0
    for c in index["cells"]:
        cost = c["explain_quota"] * cost_one
        tot_q += c["quota"]; tot_e += c["explain_quota"]
        lines.append(f"{c['gold']:>16s} -> {c['pred']:16s} {c['count']:4d} "
                     f"{c['share']:6.1%} {c['quota']:4d} {c['explain_quota']:5d} "
                     f"{cost:7.4f}")
    lines.append(f"{'TOTAL':36s} {index['meta']['n_rel_errors']:4d} {'100%':>6s} "
                 f"{tot_q:4d} {tot_e:5d} {tot_e * cost_one:7.4f}")
    return "\n".join(lines)


# ── Consumer API ─────────────────────────────────────────────────────────────

def load_index(path=INDEX_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def examples_for(index, rel_a, rel_b, k=None, offset=0):
    """Error examples for the (rel_a, rel_b) confusion, both directions,
    up to k (default: the cells' proportional quota). `offset` rotates the
    selection so different callers see different examples."""
    out = []
    for cell in index["cells"]:
        if {cell["gold"], cell["pred"]} == {rel_a, rel_b}:
            exs = cell["examples"]
            n = min(len(exs), k if k is not None else cell["quota"])
            out += [exs[(offset + i) % len(exs)] for i in range(n)]
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "train", "test"])
    ap.add_argument("--budget", type=int, default=150,
                    help="total examples allocated across cells (proportional)")
    ap.add_argument("--explain-budget", type=int, default=60,
                    help="total examples to have the LLM explain, allocated "
                         "proportionally to cell mass")
    ap.add_argument("--min-count", type=int, default=2,
                    help="drop cells with fewer errors than this (singleton "
                         "tail is likely annotation/parse noise)")
    ap.add_argument("--from-dump", action="store_true",
                    help="reuse data/stac_<split>_ddpe_preds.json instead of parsing")
    args = ap.parse_args()

    dump_file = DATA_DIR / f"stac_{args.split}_ddpe_preds.json"
    data = load_split(args.split)
    print(f"[cfg] STAC {args.split}: {len(data)} dialogues, budget={args.budget}")

    if args.from_dump:
        per_dlg = json.loads(dump_file.read_text(encoding="utf-8"))
        print(f"[dump] loaded {len(per_dlg)} dialogues from {dump_file.name}")
    else:
        per_dlg = parse_split(data, args.split)
        dump_file.write_text(json.dumps(per_dlg, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"[dump] predictions -> {dump_file.name}")

    matrix, cell_examples, n_links = harvest(data, per_dlg)
    index = build_index(matrix, cell_examples, n_links, args.budget,
                        args.explain_budget, args.split,
                        min_count=args.min_count)
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=1),
                          encoding="utf-8")

    m = index["meta"]
    print(f"\ncorrect links: {m['n_correct_links']}   relation errors: {m['n_rel_errors']}")
    print(f"cells kept: {m['n_cells_kept']}/{m['n_cells_total']} "
          f"(dropped tail: {m['tail_dropped']['cells']} cells / "
          f"{m['tail_dropped']['errors']} errors below min_count={m['min_count']})")
    print(f"\nPer-cell allocation (idx = examples kept in index, "
          f"expl = examples to LLM-explain):")
    print(cell_report(index))
    print(f"\nindex -> {INDEX_FILE}")


if __name__ == "__main__":
    main()
