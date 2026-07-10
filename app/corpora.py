"""Dataset configs + dialogue helpers for MuDiPA (extracted from DSDP)."""
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # app/ -> project root -> data/

# Root under which DISRPT corpora live, one sub-dir per corpus holding the
# original shared-task files (e.g. data/disrpt/eng.sdrt.stac/eng.sdrt.stac_test.rels).
# Override with MUDIPA_DISRPT_DIR to point at an existing DISRPT checkout.
DISRPT_DIR = Path(os.environ.get("MUDIPA_DISRPT_DIR", str(DATA_DIR / "disrpt")))

DATASETS = {
    "stac": {
        "context": "multi-party board game dialogues (Settlers of Catan). "
                   "Players negotiate trades, discuss strategy, and coordinate actions",
        "relations": "QAP, Elaboration, Continuation, Acknowledgement, Q-Elab, "
                     "Comment, Contrast, Alternation, Parallel, Correction, Result, "
                     "Explanation, Narration, Background, Conditional, Clarification_Q",
        "format": "jsonl",
        "files": {
            "test": "stac_subindex/test_subindex.json",
            "train": "stac_subindex/train_subindex.json",
            "dev": "stac_subindex/dev_subindex.json",
        },
    },
    "stac_full": {
        "context": "multi-party board game dialogues (Settlers of Catan). "
                   "Players negotiate trades, discuss strategy, and coordinate actions",
        "relations": "QAP, Elaboration, Continuation, Acknowledgement, Q-Elab, "
                     "Comment, Contrast, Alternation, Parallel, Correction, Result, "
                     "Explanation, Narration, Background, Conditional, Clarification_Q",
        "format": "json",
        "files": {
            "test": "stac/stac_test_dpa.json",
            "train": "stac/stac_linguistic_dpa.json",
        },
    },
    "molweni": {
        "context": "multi-party technical support chats (Ubuntu IRC). "
                   "Users ask questions and others provide troubleshooting help",
        "relations": "QAP, Comment, Clarification_Q, Elaboration, Continuation, "
                     "Q-Elab, Explanation, Result, Contrast, Correction, "
                     "Conditional, Alternation, Background, Narration, Parallel, "
                     "Acknowledgement",
        "format": "json",
        "files": {
            "test": "molweni/test.json",
            "train": "molweni/train.json",
            "dev": "molweni/dev.json",
        },
    },
}


def find_consecutive_groups(speakers):
    groups = []
    i = 0
    while i < len(speakers):
        j = i + 1
        while j < len(speakers) and speakers[j] == speakers[i]:
            j += 1
        if j - i > 1:
            groups.append(list(range(i, j)))
        i = j
    return groups


def merge_dialogue(edus, speakers, groups_to_merge):
    edu_to_group = {}
    for g in groups_to_merge:
        for idx in g:
            edu_to_group[idx] = tuple(g)
    merged = []
    orig_to_merged = {}
    seen_groups = set()
    merged_idx = 0
    for i in range(len(edus)):
        group = edu_to_group.get(i)
        if group and group not in seen_groups:
            seen_groups.add(group)
            combined = " ".join(edus[j] for j in group)
            merged.append((combined, speakers[i], list(group)))
            for j in group:
                orig_to_merged[j] = merged_idx
            merged_idx += 1
        elif not group:
            merged.append((edus[i], speakers[i], [i]))
            orig_to_merged[i] = merged_idx
            merged_idx += 1
    return merged, orig_to_merged


GEVAL_SCALE = """Rate how well the given lines form a coherent sub-dialogue on this scale:
1 - No thread: the lines are unrelated.
2 - Same topic but no flow: each line stands alone.
3 - Loose thread: thematic progression but indirect links.
4 - Clear thread: each line clearly responds to what came before.
5 - Perfect thread: unmistakable continuation."""


def get_prompts(dataset_name):
    """Generate system prompts for merge and pair evaluation."""
    cfg = DATASETS[dataset_name]
    ctx = cfg["context"]
    rels = cfg["relations"]
    rel_defs = (
        "For context, discourse relations in this domain include: "
        f"{rels}. "
        "You do not need to classify the relation type -- just judge "
        "whether a real conversational link exists."
    )
    system_merge = (
        f"You are reading {ctx}.\n\n"
        "Sometimes a speaker's turn is split into multiple lines, but they are "
        "really saying one thing. For each pair of consecutive lines by the same "
        "speaker below, tell me: is the second line a continuation of what they "
        "were already saying, or are they starting a new, independent thought?\n\n"
        f"{rel_defs}\n\n"
        "Respond in JSON:\n"
        "{\"<pair_id>\": {\"from\": <id>, \"to\": <id>, \"connected\": true/false, "
        "\"reason\": \"<brief>\"}, ...}"
    )
    system_pair = (
        f"You are reading {ctx}.\n\n"
        "I will show you a conversation and pick out a sequence of lines. "
        "Your job: do these lines form a coherent sub-dialogue?\n\n"
        f"{rel_defs}\n\n"
        f"{GEVAL_SCALE}\n\n"
        "Answer with just the number (1-5)."
    )
    return system_merge, system_pair


# ── Pluggable dataset loaders ────────────────────────────────────────────────
# A dataset config declares a "loader" (default "native"); load_records() dispatches
# on it. A loader takes (cfg, split) and returns a list of MuDiPA dialogue records:
#     {"id": str, "edus": [{"speaker", "text", ...}], "relations": [{"x","y","type"}]}
# Adding support for a new corpus family = registering one loader here -- callers
# (app.py) go through load_records() and never touch format-specific code.

DATASET_LOADERS = {}


def register_loader(name):
    def deco(fn):
        DATASET_LOADERS[name] = fn
        return fn
    return deco


def load_records(dataset, split):
    """Load dialogue records for (dataset, split), dispatching on the config's loader."""
    cfg = DATASETS[dataset]
    loader = DATASET_LOADERS.get(cfg.get("loader", "native"))
    if loader is None:
        raise ValueError(f"dataset {dataset!r}: unknown loader {cfg.get('loader')!r} "
                         f"(known: {sorted(DATASET_LOADERS)})")
    return loader(cfg, split)


@register_loader("native")
def _load_native(cfg, split):
    """Native MuDiPA json/jsonl: records already in the internal schema."""
    path = DATA_DIR / cfg["files"][split]
    fmt = cfg.get("format", "json")
    with open(path, encoding="utf-8") as f:
        if fmt == "jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def disrpt_dataset(corpus, context, relations, splits=("train", "dev", "test")):
    """Build a DATASETS entry for a DISRPT corpus. `files` maps split -> the file
    *prefix* (the loader appends .rels/.conllu); the corpus dir lives under DISRPT_DIR."""
    return {
        "loader": "disrpt",
        "corpus": corpus,
        "context": context,
        "relations": relations,
        "files": {sp: f"{corpus}_{sp}" for sp in splits},
    }


# draddp: DraDDP-style multimodal multi-party dialogue discourse parsing (Friends,
# STAC's 16 SDRT relations). Built from real Friends dialogues via MELD
# (arXiv:1810.02508): per-utterance speaker + timestamps (-> clip) from
# data/draddp/<split>_sent_emo.csv, per-EDU video/audio clips in data/draddp/media/,
# with discourse arcs generated by the DDPE parser on the text (generate_meld_parse.py
# -> data/draddp/parsed_<split>.json). Arcs are parser-predicted, not gold.
DATASETS["draddp"] = {
    "loader": "meld",
    "context": "multi-party English TV-series conversations (Friends; DraDDP via "
               "MELD); characters joke, argue, and react across everyday scenes",
    "relations": DATASETS["stac"]["relations"],
    "files": {"train": "draddp/train_sent_emo.csv",
              "dev": "draddp/dev_sent_emo.csv",
              "test": "draddp/test_sent_emo.csv"},
}

# MODDP: real Multimodal Dialogue Discourse Parsing corpus (Chinese TV-drama
# conversations). Ships annotations (text + SDRT-style relations) + video filename
# refs; the video files and per-EDU clips are NOT included, so it loads as a
# text+relations dataset (the media panel stays hidden until media is provided).
DATASETS["moddp"] = {
    "context": "multi-party Chinese TV-drama conversations (Multimodal Dialogue "
               "Discourse Parsing); characters argue, explain, and react",
    "relations": DATASETS["stac"]["relations"],
    "format": "json",
    "files": {"train": "moddp/train.json", "dev": "moddp/dev.json",
              "test": "moddp/test.json"},
}

# mmdemo retired: the multimodal frame-view "mode" it demonstrated is now carried by
# draddp_demo on real Friends media (see meld_adapter media resolution).

# sample: tiny synthetic demo corpus shipped with the repo (data/sample/) — text
# dialogues stratified by length plus one multimodal audio dialogue. Lets anyone try
# the tool with no licensed data; media (demo_audio.wav) lives in data/sample/media/.
DATASETS["sample"] = {
    "context": "small synthetic multi-party dialogues for trying the tool",
    "relations": DATASETS["stac"]["relations"],
    "loader": "native", "format": "jsonl",
    "files": {"test": "sample/demo.jsonl"},
}

# eng.sdrt.stac: STAC in the DISRPT SDRT format (real DISRPT-2025 shared-task files
# under data/disrpt/), read straight from the original .rels/.conllu. The relation
# is taken from `orig_label` (SDRT names), not the coarse DISRPT `label`.
DATASETS["eng.sdrt.stac"] = disrpt_dataset(
    "eng.sdrt.stac",
    context=DATASETS["stac"]["context"],
    relations=DATASETS["stac"]["relations"],
)

# eng.sdrt.msdc: Minecraft Structured Dialogue Corpus (SDRT), DISRPT-2025 format.
# Multi-party collaborative-building dialogues; same SDRT relation inventory.
DATASETS["eng.sdrt.msdc"] = disrpt_dataset(
    "eng.sdrt.msdc",
    context="multi-party collaborative Minecraft building dialogues; players "
            "instruct, clarify, and coordinate to build a target structure",
    relations=DATASETS["stac"]["relations"],
)

def _is_redacted(rels_path):
    """True if a .rels file's unit text is underscore-masked (licensing redaction,
    e.g. PTB/WSJ corpora). Cheap peek at the first data row's unit1_txt column."""
    try:
        with open(rels_path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            row = f.readline().rstrip("\n").split("\t")
        idx = header.index("unit1_txt") if "unit1_txt" in header else 5
        if idx >= len(row):
            return False
        chars = row[idx].replace(" ", "")
        return bool(chars) and sum(c == "_" for c in chars) / len(chars) > 0.6
    except Exception:
        return False


def _autoregister_disrpt():
    """Register every DISRPT corpus present under DISRPT_DIR (a dir per corpus with
    <corpus>_<split>.rels files), so all copied shared-task corpora — with their
    ORIGINAL gold (`orig_label` relation + `dir` structure) — appear as datasets.
    Corpora explicitly registered above (custom context) are left untouched. SDRT
    corpora are dialogue; RST/PDTB/DEP are mostly monologue (they load with gold, but
    are not dialogue). Only splits whose .rels exists are exposed."""
    if not DISRPT_DIR.exists():
        return
    for d in sorted(DISRPT_DIR.iterdir()):
        corpus = d.name
        if not d.is_dir() or corpus in DATASETS:
            continue
        splits = {sp: f"{corpus}_{sp}" for sp in ("train", "dev", "test")
                  if (d / f"{corpus}_{sp}.rels").exists()}
        if not splits:
            continue
        # Skip corpora whose text is REDACTED (underscore-masked for licensing, e.g.
        # PTB/WSJ-derived eng.pdtb.pdtb): they load but are unreadable/unannotatable.
        if _is_redacted(d / f"{corpus}_{next(iter(splits))}.rels"):
            continue
        framework = corpus.split(".")[1] if corpus.count(".") >= 2 else "discourse"
        DATASETS[corpus] = {
            "loader": "disrpt",
            "corpus": corpus,
            "context": f"DISRPT-2025 {framework.upper()} corpus '{corpus}'",
            "relations": (DATASETS["stac"]["relations"] if framework == "sdrt"
                          else f"{framework.upper()} discourse relations"),
            "files": splits,
        }


_autoregister_disrpt()

# Import registers the "disrpt" loader (see disrpt_adapter.py). Done last so the
# registry helpers above already exist when the adapter imports them back.
from disrpt_adapter import load_disrpt  # noqa: E402,F401
from meld_adapter import load_meld  # noqa: E402,F401
