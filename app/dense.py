"""
Dense arc embedding backends for within-cell confusion retrieval.

Two backends, both returning L2-normalised vectors (cosine = dot product):

  * ddpe  — the parser's OWN latent space (suggestion_engine /ddpe/embed): the hidden
            state from which DDPE generates its prediction. "Similar" here means
            "the parser treats these arcs alike", i.e. errs on them alike — the
            right space for retrieving relevant error examples. Needs the full
            dialogue up to the target EDU.
  * st    — general sentence-transformers (all-MiniLM-L6-v2) over the arc text.
            Topic-biased, but used only WITHIN a fixed confusion cell where the
            relation-pair is already controlled, so it ranks situational
            similarity. Fallback when the DDPE server is down / no full dialogue.

Backend chosen by env CONF_DENSE = "ddpe" (default) | "st" | "none".
"""
import os

import numpy as np
import requests

SUGGESTION_ENGINE_URL = os.environ.get("SUGGESTION_ENGINE_URL", "http://127.0.0.1:8092")
_ST = None


def backend_name():
    return (os.environ.get("CONF_DENSE") or "ddpe").lower()


# ── DDPE latent space ────────────────────────────────────────────────────────

def ddpe_embed(items, dataset="stac", url=SUGGESTION_ENGINE_URL, batch=16, timeout=120):
    """items: [{'edus':[{speaker,text}...], 'source':int, 'target':int}]. The
    embedding is taken at the arc's relation-decision point '[t]–>[p]:', so both
    source and target matter. Returns [N,D] float32."""
    vecs = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        r = requests.post(f"{url}/ddpe/embed",
                          json={"items": chunk, "dataset": dataset}, timeout=timeout)
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    return np.asarray(vecs, dtype=np.float32)


def ddpe_available(url=SUGGESTION_ENGINE_URL):
    try:
        return requests.get(f"{url}/health", timeout=5).ok
    except Exception:
        return False


# ── sentence-transformers ────────────────────────────────────────────────────

def _st_model():
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST


def st_embed(texts):
    """texts: list[str] -> [N,D] float32, L2-normalised."""
    return np.asarray(
        _st_model().encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)


def arc_text(src_speaker, src_text, tgt_speaker, tgt_text):
    return f"{src_speaker}: {src_text} -> {tgt_speaker}: {tgt_text}"


# ── unified helpers ──────────────────────────────────────────────────────────

def embed_examples(examples, full_edus_by_dlg, dataset="stac", backend=None):
    """Embed index examples. `full_edus_by_dlg[dlg_id]` = [{speaker,text}...] for
    the DDPE backend (needs dialogue context up to tgt); ST uses only arc text."""
    backend = backend or backend_name()
    if backend == "none":
        return None, "none"
    if backend == "ddpe":
        items = [{"edus": full_edus_by_dlg[e["dlg_id"]],
                  "source": e["src"], "target": e["tgt"]}
                 for e in examples]
        return ddpe_embed(items, dataset=dataset), "ddpe"
    texts = [arc_text(e["src_speaker"], e["src_text"], e["tgt_speaker"], e["tgt_text"])
             for e in examples]
    return st_embed(texts), "st"


def embed_query(src_speaker, src_text, tgt_speaker, tgt_text,
                full_edus=None, source=None, target=None, dataset="stac", backend=None):
    """One query vector, matching the backend the examples were embedded with.
    DDPE needs full_edus + source + target (relation-decision point '[t]–>[p]:');
    ST needs only the arc utterances."""
    backend = backend or backend_name()
    if backend == "ddpe" and full_edus is not None and source is not None and target is not None:
        return ddpe_embed([{"edus": full_edus, "source": source, "target": target}],
                          dataset=dataset)[0]
    if backend == "none":
        return None
    return st_embed([arc_text(src_speaker, src_text, tgt_speaker, tgt_text)])[0]
