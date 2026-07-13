# MuDiPA — System Reference

MuDiPA is a web-based **human-in-the-loop** platform for **discourse graph** annotation in
multi-party dialogues. Annotators build the graph over elementary discourse units (EDUs) —
attachment **arcs** and **relation** labels — assisted by two **pluggable AI backends** whose
suggestions and rationales the human inspects, compares, and always overrides.

---

## 1. Architecture

The interface is decoupled from any specific AI backend behind **two pluggable slots**:

```
  Browser SPA  ◀─HTTP─▶  Flask backend (app.py :5050)  ─────▶  SUGGESTION ENGINE
  single-file React                 │                          (discourse parser w/
  infinite pan-&-zoom               │                           scoring API)
  SVG canvas                        │
                                    └────────────────────────▶  EXPLANATION ENGINE
                                    │                            (LLM w/ rationale API)
                                    ▼                             
                          data/ (datasets · media · annotations · mudipa.db)
```

- **Suggestion engine** — any discourse parser exposing a **link-scoring** and
  **relation-scoring** API. It *drafts* candidate arcs with **confidence scores**. 
  Concrete instance: **DDPE** (LLaMA-3 + LoRA, STAC/Molweni), `suggestion_engine.py` on :8092.
- **Explanation engine** — any LLM that produces on-demand **natural-language rationales** for a
  candidate link and relation. Concrete instance: **Claude** (local CLI via OAuth, or Anthropic SDK).
- **Backend** — `app.py` (Flask): serves the SPA, resolves datasets, persists annotations, and
  *proxies* both engines. It never loads a GPU model itself.



---

## 2. Pluggable engine architecture 

`engines.py` defines a pure `Engine` base + `Registry`. An engine advertises **capabilities**
and implements the matching methods:

| Capability | Method | Slot |
|---|---|---|
| `can_score_links` | `score_links(edus, target, dataset)` → ranked parents | suggestion |
| `can_score_rel`   | `score_rel(edus, src, tgt, dataset)` → ranked relations | suggestion |
| `can_score_pair`  | `score_pair(edus, src, tgt, dataset[, target_text])` → attach score | suggestion (saliency) |
| `can_explain`     | `explain(edus, src, tgt, dataset, …)` → link/relation/contrastive | explanation |
| `needs_activation`| `activate(dataset)` → warm-up / readiness | — |

Concrete engines (registered at import): **`DdpeEngine`** (suggestion: links, relations, attach
score, + explain), **`ClaudeEngine`** (explanation). `DediscoEngine`/`ZeroshotEngine` exist as
inactive fallbacks. Discovery: `GET /api/engines`; activation: `POST /api/engine/activate`.
**Adding a backend = one subclass + `register()`** — nothing in the routes changes.

---

## 3. The annotation workflow

1. **Draft.** The suggestion engine scores candidate parents for the current EDU
   (`POST /api/suggest/links`, right-frontier-renormalized) and relations for an arc
   (`POST /api/suggest/relation`, structural rules applied).
2. **Uncertainty routing.** Candidates are routed by confidence: high-confidence (typically
   short-distance) arcs surface as **proposal chips** — dashed annotations pinned beside the
   graph that the human accepts/rejects; the rest are ranked into a confidence-ordered review
   queue (`POST /api/review_queue`). This focuses human attention on the genuinely hard,
   long-distance attachments.
3. **Explain on demand.** For any candidate, the explanation engine returns a **three-tab
   rationale** — *link existence*, *relation choice*, *contrastive comparison*
   (`POST /api/suggest/incremental`, Claude EGM) — plus a **faithful occlusion saliency** over
   the parser's attach score (`POST /api/explain/arc`, `score_pair` per token; DDPE-grounded).
4. **Decide & persist.** The human sets every arc and label; annotations save per participant
   (`POST /api/annotation`) as JSON + a timestamped SQLite row.

---

## 4. Discourse-specific features  

- **Relation codebook.** Inline definitions (and examples) shown while labelling. The codebook is
  **per-dataset**: each dataset exposes its own relation **inventory**, and definitions come from
  a **cross-framework glossary** (`relation_glossary.py`) covering SDRT, RST/eRST, PDTB 3.0,
  SciDTB, ISO 24617-8 — composed from the original taxonomies. `GET /api/relation_defs?dataset=`
  → `{framework, inventory, relations(defs)}`.
- **Right-Frontier Constraint enforcement.** `relation_rules.right_frontier` restricts and
  renormalizes valid attachment points; `filter_relation_scores` zeros structurally-impossible
  relations before ranking.
- **Thread detection.** EDUs are grouped into conversational sub-threads
  (`POST /api/suggest/threads`, `POST /api/discourse/threads`).
- **Multimodal panel with per-EDU seeking** (§5.3).
- **Layouts & canvas.** Infinite pan-and-zoom **SVG canvas**; several layout arrangements for long
  or heavily threaded dialogues.
- **Export.** The annotated graph exports as **JSON** (or **STAC** / **Glozz**),
  `POST /api/export`.

---

## 5. Datasets & frameworks 

### 5.1 Dataset loader registry
`corpora.py` decouples datasets behind a loader registry (`register_loader`, `load_records`).
Loaders: **`native`** (native json/jsonl), **`disrpt`** (`.rels`/`.conllu`), **`meld`** (MELD → the
`draddp` corpus). All datasets are picked from one **faceted selector** (`DatasetPicker`): native
datasets as buttons + DISRPT corpora grouped **language → formalism → corpus**.

### 5.2 Multilingual — DISRPT
Every DISRPT-2025 corpus under `data/disrpt/` **auto-registers** (`disrpt_adapter.py`), keeping
its **original gold** (`orig_label`) and its **native relation inventory** (RST/PDTB/DEP/ISO/
SDRT — no forced mapping to SDRT). Licence-redacted corpora (PTB/WSJ, underscore-masked text) are
excluded; `erst` is folded into `rst`; undirected relations default to `arg1→arg2`.

### 5.3 Multimodal 
Real *Friends* dialogues from **MELD**: per-utterance **speaker + timestamps → per-EDU `clip`**,
per-utterance **video/audio → per-EDU `clip_src`**. Each EDU's own clip (video+audio, or audio)
renders **on its graph node**, aligned to its timestamp — the paper's *"multimodal panel with
per-EDU timestamp seeking,"* taken to per-EDU segments. Dialogue-level **waveform peaks** are
precomputed (`precompute_media.py`; ffmpeg via imageio-ffmpeg for non-WAV/video).

---

## 6. Data model & persistence

```jsonc
{ "id": "dia1",
  "edus": [ { "speaker", "text",
              "clip": [start_s, end_s],           // offset in the dialogue media
              "clip_src": "/media/<ds>/clips/…",  // this EDU's own segment
              "clip_type": "video"|"audio" }, … ],
  "relations": [ { "x": src, "y": tgt, "type": "<label>" }, … ],
  "media": { "type", "src", "peaks" } }
```
Arc `x→y`: `x` source/parent (earlier), `y` dependent (later). Annotation
`{arcs, threads, notes}` → `data/annotations/<pid>_<dataset>_<split>.json` + SQLite `mudipa.db`
(users, annotations, events). Media served from `data/<dataset>/media/` with HTTP range.


---

