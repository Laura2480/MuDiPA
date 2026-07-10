"""
Faceted retrieval over the confusion-example index (build_confusion_index.py).

Two-stage retriever, matching the criterion "same relation-pair, then ranked
by similarity":
  1. HARD facet — the confusion cell (gold, pred). Retrieval never crosses
     cells; the matrix itself is the first-stage index.
  2. Within a cell — rank error examples by discourse-functional similarity to
     the query arc (arc_sim.py): speaker structure, attachment distance, target
     dialogue act, and cue markers dominate; topic is a small tie-break. NO
     embedding model — this is deliberately not topic-similarity and runs with
     no heavy deps.

LLM-generated contrastive rationales (build_error_explanations.py) are attached
to each example as `ex["why"]` when available.

API:
    rag = ConfusionRAG()
    rag.cell(gold, pred)                         # the cell's examples (+why)
    rag.retrieve_in_cell(gold, pred, query_sig, k)   # ranked within one cell
    rag.pred_column(pred)                        # [(gold, share, cell), ...] desc
"""
import argparse
import json
from pathlib import Path

import numpy as np

from build_confusion_index import INDEX_FILE, load_index
from arc_sim import arc_signature, signature_of_example, arc_similarity

ROOT      = Path(__file__).parent
EXPL_FILE = ROOT / "data" / "stac_error_explanations.json"
DENSE_FILE = ROOT / "data" / "stac_confusion_index_dense.npz"

# within-cell blend: final = ALPHA*structural + (1-ALPHA)*dense_cosine
ALPHA = 0.55


def _load_expl():
    if EXPL_FILE.exists():
        return json.loads(EXPL_FILE.read_text(encoding="utf-8"))
    return {}


def _load_dense():
    """key -> L2-normalised vector, plus the backend name. {} if absent."""
    if not DENSE_FILE.exists():
        return {}, None
    z = np.load(DENSE_FILE, allow_pickle=False)
    vecs, keys = z["vecs"], [str(k) for k in z["keys"]]
    return {k: vecs[i] for i, k in enumerate(keys)}, str(z["backend"])


class ConfusionRAG:
    def __init__(self, index_path=INDEX_FILE, alpha=ALPHA):
        self.index = load_index(index_path)
        self.alpha = alpha
        expl = _load_expl()
        dense, self.dense_backend = _load_dense()
        # index cells by (gold, pred); precompute example signatures + attach
        # why-rationale and dense vector
        self.cells = {}
        for cell in self.index["cells"]:
            exs = []
            for ex in cell["examples"]:
                key = f"{ex['dlg_id']}:{ex['src']}:{ex['tgt']}"
                e = {**ex, "share": cell["share"], "_sig": signature_of_example(ex),
                     "_vec": dense.get(key)}
                if key in expl:
                    e["why"] = {k: expl[key][k]
                                for k in ("reasoning", "why_not", "cue")
                                if k in expl[key]}
                exs.append(e)
            self.cells[(cell["gold"], cell["pred"])] = {
                "share": cell["share"], "count": cell["count"], "examples": exs}

    def cell(self, gold, pred):
        return self.cells.get((gold, pred))

    def embed_query(self, src_speaker, src_text, tgt_speaker, tgt_text,
                    full_edus=None, source=None, target=None, dataset="stac"):
        """Query vector in the SAME space the example vectors use, or None if no
        dense index is loaded / embedding fails (→ structural-only ranking).
        DDPE backend needs full_edus + source + target (relation-decision point)."""
        if not self.dense_backend:
            return None
        try:
            import dense as D
            return D.embed_query(src_speaker, src_text, tgt_speaker, tgt_text,
                                 full_edus=full_edus, source=source, target=target,
                                 dataset=dataset, backend=self.dense_backend)
        except Exception:
            return None

    def pred_column(self, pred):
        """All confusion cells where the parser predicted `pred`, i.e. the
        true-relation distribution given that prediction, most frequent first."""
        col = [(g, c["share"], c) for (g, p), c in self.cells.items()
               if p == pred and g != pred]
        col.sort(key=lambda t: -t[1])
        return col

    def most_confusable(self, rel):
        """The relation `rel` is most often confused WITH on this corpus, summing
        both directions (rel mislabelled as X, and X mislabelled as rel). Returns
        {partner, count, share, cue, direction} or None. Use as the pedagogically
        strongest contrastive when the LLM agrees `rel` is correct: it warns
        against the mistake actually made on `rel`, not a random runner-up."""
        agg = {}   # partner -> {count, as_pred, as_gold}
        for (g, p), c in self.cells.items():
            if g == rel and p != rel:          # rel (gold) mislabelled as p
                a = agg.setdefault(p, {"count": 0, "as_pred": 0, "as_gold": 0})
                a["count"] += c["count"]; a["as_pred"] += c["count"]
            elif p == rel and g != rel:        # g mislabelled as rel (pred)
                a = agg.setdefault(g, {"count": 0, "as_pred": 0, "as_gold": 0})
                a["count"] += c["count"]; a["as_gold"] += c["count"]
        if not agg:
            return None
        partner = max(agg, key=lambda x: agg[x]["count"])
        info = agg[partner]
        n_err = self.index["meta"].get("n_rel_errors", 0) or 1
        # cue: prefer the cell in the dominant direction, fall back to the other
        cue = None
        for g, p in (((rel, partner) if info["as_pred"] >= info["as_gold"]
                      else (partner, rel)),
                     ((partner, rel) if info["as_pred"] >= info["as_gold"]
                      else (rel, partner))):
            c = self.cells.get((g, p))
            if c:
                for e in c["examples"]:
                    if e.get("why", {}).get("cue"):
                        cue = e["why"]["cue"]; break
            if cue:
                break
        return {"partner": partner, "count": info["count"],
                "share": info["count"] / n_err, "cue": cue,
                "direction": "as_pred" if info["as_pred"] >= info["as_gold"] else "as_gold"}

    def retrieve_in_cell(self, gold, pred, query_sig, k=2, query_vec=None):
        """Top-k error examples of cell (gold, pred), ranked by the LEAST
        handcrafted signal available for the query arc:

          * dense_backend == 'ddpe'  -> pure cosine in the PARSER'S latent space.
            The parser is trained to attend to speaker/distance/act/cues to
            decide the arc, so this cosine IS learned discourse similarity — no
            hand-tuned weights, no mixing constant.
          * dense_backend == 'st'    -> a heuristic FALLBACK: raw MiniLM cosine is
            topic-biased (wheat/sheep), so it is corrected by the structural
            score (arc_sim). This is the only path with hand-picked weights, and
            it exists only until the parser-latent backend is built.
          * no dense                 -> structural only (arc_sim) fallback.
        """
        c = self.cells.get((gold, pred))
        if not c:
            return []
        qv = np.asarray(query_vec, dtype=np.float32) if query_vec is not None else None
        dense_ok = qv is not None and self.dense_backend
        scored = []
        for e in c["examples"]:
            if dense_ok and e.get("_vec") is not None:
                cos = (float(np.dot(qv, e["_vec"])) + 1) / 2
                if self.dense_backend == "ddpe":
                    score = cos                                  # learned, no handcraft
                else:                                            # 'st' heuristic fallback
                    score = self.alpha * arc_similarity(query_sig, e["_sig"]) \
                            + (1 - self.alpha) * cos
            else:
                score = arc_similarity(query_sig, e["_sig"])     # structural fallback
            scored.append((round(score, 4), e))
        scored.sort(key=lambda t: -t[0])
        return scored[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="show the true-relation column for a parser prediction")
    ap.add_argument("--query", help="'spk: text -> spk: text' arc to rank against")
    ap.add_argument("--gold", help="gold side of the cell to retrieve from")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    rag = ConfusionRAG()
    if args.pred and not args.query:
        print(f"When parser predicts '{args.pred}', true relation is:")
        for g, share, c in rag.pred_column(args.pred):
            print(f"  {g:16s} {share:6.1%}  ({c['count']} errors)")
        return
    if args.query:
        left, right = args.query.split("->", 1)
        s_spk, s_txt = (left.split(":", 1) + [""])[:2]
        t_spk, t_txt = (right.split(":", 1) + [""])[:2]
        sig = arc_signature(s_spk.strip(), s_txt.strip(),
                            t_spk.strip(), t_txt.strip(), distance=1)
        pred = args.pred or "Comment"
        golds = [args.gold] if args.gold else [g for g, _, _ in rag.pred_column(pred)[:3]]
        for g in golds:
            print(f"\ncell  gold {g}  ~  pred {pred}:")
            for score, ex in rag.retrieve_in_cell(g, pred, sig, k=args.k):
                print(f"  {score:.3f}  {ex['src_speaker']}: {ex['src_text']!r} -> "
                      f"{ex['tgt_speaker']}: {ex['tgt_text']!r}")


if __name__ == "__main__":
    main()
