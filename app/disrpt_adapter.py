"""DISRPT shared-task format -> MuDiPA internal schema adapter.

Reads the *original* DISRPT ``.rels`` (+ ``.conllu`` for speaker/ordering) files
without any pre-conversion and yields MuDiPA dialogue records:

    {"id": doc_id,
     "edus": [{"speaker": str, "text": str}, ...],   # ordered by token offset
     "relations": [{"x": src_edu, "y": tgt_edu, "type": label}, ...]}

DISRPT ``.rels`` is a TSV with a header row and one *discourse relation* per line,
each relation connecting two units identified by token spans (per-document,
1-based token ids, matching the ``.conllu`` numbering). We reconstruct each
document's EDU list by collecting the distinct units it mentions, ordering them by
token offset, and mapping each relation's ``dir`` field (``1>2`` / ``1<2``) onto a
source->target arc (the arrow points from head/source to dependent/target, matching
MuDiPA's ``x`` = source, ``y`` = target convention).

Columns are resolved by *header name* (stable across DISRPT editions even as
positions drift): ``doc``, ``unit1_toks``, ``unit2_toks``, ``unit1_txt``,
``unit2_txt``, ``dir``, ``label`` (falling back to ``orig_label``). If a name is
absent we fall back to the official ``disrpt_eval_2024`` positions
(doc=0, unit1_toks=1, unit2_toks=2, unit1_txt=5, unit2_txt=6, dir=-4, label=-1).

Limitation: EDUs are reconstructed from units that appear in ``.rels``; a segment
with no relations at all is not surfaced. This is fine for annotation of the
connected structure; full segmentation would additionally read the ``.tok`` BIO tags.
"""
from pathlib import Path

from corpora import register_loader, DISRPT_DIR

# Preferred header names -> positional fallback (official disrpt_eval_2024 indices).
_FIELD_FALLBACK = {
    "doc": 0, "unit1_toks": 1, "unit2_toks": 2,
    "unit1_txt": 5, "unit2_txt": 6, "dir": -4, "label": -2,
}
# Alternate header names accepted for a field, in priority order. For the relation
# we want the corpus-native SDRT label (`orig_label` = "Continuation", "Question_
# answer_pair", ...), NOT the DISRPT-2025 coarse `label` ("conjunction", "query").
_FIELD_ALIASES = {"label": ("orig_label", "label")}


def _resolve_cols(header):
    """Map each logical field -> column index, by header name with positional fallback."""
    idx = {name: i for i, name in enumerate(header)}
    cols = {}
    for field, fallback in _FIELD_FALLBACK.items():
        chosen = None
        for alias in _FIELD_ALIASES.get(field, (field,)):
            if alias in idx:
                chosen = idx[alias]
                break
        cols[field] = chosen if chosen is not None else fallback
    return cols


def _toks(span):
    """Expand a token-span string ("2-6" / "1" / "2-6,9-11") into a sorted list of ids."""
    out = []
    for part in str(span).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")[:2]
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(out)


def _parse_rels(path):
    """Return (header, rows) where rows are raw column lists (header excluded)."""
    rows = []
    head = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").replace("\r", "")
            if not line.strip():
                continue
            cols = line.split("\t")
            if head is None:
                head = cols
                continue
            rows.append(cols)
    return head or [], rows


def _conllu_speakers(path):
    """Map {doc_id: {global_token_id: speaker}} from a .conllu file.

    Speaker is read from ``# speaker = X`` sentence metadata (as in DISRPT SDRT
    corpora); token ids restart at 1 per ``# newdoc``, matching .rels numbering.
    Returns an empty mapping if the file is missing."""
    docs = {}
    cur = None
    gid = 0
    spk = ""
    if not Path(path).exists():
        return docs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            low = line.lower()
            if low.startswith("# newdoc"):
                cur = line.split("=", 1)[1].strip() if "=" in line else None
                docs[cur] = {}
                gid = 0
                spk = ""
                continue
            if low.startswith("# speaker"):
                spk = line.split("=", 1)[1].strip() if "=" in line else ""
                continue
            if line.startswith("#") or not line.strip():
                continue
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:   # skip multiword / empty nodes
                continue
            gid += 1
            if cur is not None:
                docs[cur][gid] = spk
    return docs


@register_loader("disrpt")
def load_disrpt(cfg, split):
    """Load a DISRPT corpus split into MuDiPA dialogue records (one per document)."""
    corpus = cfg["corpus"]
    prefix = cfg["files"][split]                     # e.g. "eng.sdrt.stac_test"
    # NB: corpus names contain dots -> don't use Path.with_suffix (it would treat
    # ".stac_test" as the extension). Append the extension to the string instead.
    base = DISRPT_DIR / corpus / prefix
    rels_path = Path(f"{base}.rels")
    if not rels_path.exists():
        raise FileNotFoundError(
            f"DISRPT file not found: {rels_path}\n"
            f"Place the original {corpus} shared-task files under {DISRPT_DIR / corpus}/ "
            f"(or set MUDIPA_DISRPT_DIR). Expected {prefix}.rels (+ {prefix}.conllu).")

    speakers_by_doc = _conllu_speakers(Path(f"{base}.conllu"))

    header, raw_rows = _parse_rels(rels_path)
    col = _resolve_cols(header)
    need = max(col["unit1_txt"], col["unit2_txt"], col["unit1_toks"],
               col["unit2_toks"], col["doc"]) if header else 6

    # doc_id -> {span_str: {"start": int, "text": str}} and list of (u1span,u2span,dir,label)
    units = {}
    arcs = {}
    doc_order = []
    for c in raw_rows:
        if len(c) <= need:
            continue
        doc = c[col["doc"]]
        if doc not in units:
            units[doc] = {}
            arcs[doc] = []
            doc_order.append(doc)
        for span_col, txt_col in (("unit1_toks", "unit1_txt"), ("unit2_toks", "unit2_txt")):
            span = c[col[span_col]]
            if span and span not in units[doc]:
                tks = _toks(span)
                units[doc][span] = {"start": tks[0] if tks else 10**9,
                                    "text": c[col[txt_col]].strip()}
        arcs[doc].append((c[col["unit1_toks"]], c[col["unit2_toks"]],
                          c[col["dir"]], c[col["label"]]))

    records = []
    for doc in doc_order:
        # order EDUs by first-token offset -> index
        ordered = sorted(units[doc].items(), key=lambda kv: (kv[1]["start"], kv[0]))
        span_to_idx = {span: i for i, (span, _) in enumerate(ordered)}
        spk_map = speakers_by_doc.get(doc, {})
        edus = [{"speaker": spk_map.get(info["start"], ""), "text": info["text"]}
                for _, info in ordered]

        relations = []
        for u1, u2, direction, label in arcs[doc]:
            if u1 not in span_to_idx or u2 not in span_to_idx:
                continue
            i1, i2 = span_to_idx[u1], span_to_idx[u2]
            if "<" in direction:            # 1<2: unit2 is head/source
                x, y = i2, i1
            else:                           # 1>2, or undirected ("_"/"" in some PDTB
                x, y = i1, i2               # corpora): default unit1 -> unit2 (arg1->arg2)
            if x == y:
                continue
            relations.append({"x": x, "y": y, "type": (label or "").strip()})

        records.append({"id": doc, "edus": edus, "relations": relations})
    return records
