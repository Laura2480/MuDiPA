"""MELD (Multimodal EmotionLines Dataset) -> MuDiPA schema adapter.

MELD (Poria et al., 2019; arXiv:1810.02508) is the *Friends* TV-series multimodal
dataset: per-utterance transcripts, speakers, emotion/sentiment, episode
timestamps, and one video clip per utterance (``diaX_uttY.mp4``). It is the public,
real multimodal Friends substrate for discourse annotation — the DraDDP task uses
the same source, and MELD supplies the media DraDDP does not release.

Utterances are grouped by ``Dialogue_ID`` into MuDiPA dialogues; each utterance becomes
an EDU (speaker, text) with a ``clip`` = cumulative ``[start, end]`` offsets within a
*per-dialogue* video (the concatenation of that dialogue's utterance clips). MELD has
no discourse relations (it is emotion, not discourse) -> ``relations: []``, to be
annotated in MuDiPA (or predicted by DDPE/Claude). Media points at
``data/meld/media/dia<id>.mp4``; build those clips (+ waveform/keyframes) with
precompute — until the file exists the dialogue degrades to text-only.
"""
import csv
import json

from corpora import register_loader, DATA_DIR


def _sec(t):
    """'hh:mm:ss,ms' -> seconds (float); tolerant of blanks/quotes."""
    t = str(t or "").strip().strip('"')
    if not t:
        return 0.0
    hms, _, ms = t.partition(",")
    parts = hms.split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        else:
            return 0.0
    except ValueError:
        return 0.0
    return h * 3600 + m * 60 + s + (int(ms) / 1000.0 if ms.isdigit() else 0.0)


@register_loader("meld")
def load_meld(cfg, split):
    """Load a MELD split (<split>_sent_emo.csv) into MuDiPA dialogue records."""
    path = DATA_DIR / cfg["files"][split]
    if not path.exists():
        raise FileNotFoundError(
            f"MELD csv not found: {path}\nDownload the MELD annotation CSVs into "
            f"data/meld/ (declare-lab/MELD: data/MELD/<split>_sent_emo.csv).")

    # Optional parser-generated arcs (generate_meld_parse.py) — a DEMO overlay since
    # MELD has no gold discourse structure. {dialogue_id: [{x,y,type,origin}]}.
    parsed = {}
    parsed_path = path.parent / f"parsed_{split}.json"
    if parsed_path.exists():
        try:
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        except Exception:
            parsed = {}

    media_dir = path.parent / "media"

    def _media_for(did):
        """Point at whatever media file exists for this dialogue (built by
        build_meld_media.py): video mp4 preferred, else audio wav. Falls back to a
        video ref that get_dialogue drops when the file is absent (text-only)."""
        if (media_dir / f"dia{did}.mp4").exists():
            return {"type": "video", "src": f"dia{did}.mp4"}
        if (media_dir / f"dia{did}.wav").exists():
            return {"type": "audio", "src": f"dia{did}.wav"}
        return {"type": "video", "src": f"dia{did}.mp4"}

    dialogues = {}
    order = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                did = int(row["Dialogue_ID"])
                uid = int(row["Utterance_ID"])
            except (KeyError, ValueError):
                continue
            if did not in dialogues:
                dialogues[did] = []
                order.append(did)
            dialogues[did].append((uid, row))

    records = []
    for did in order:
        rows = sorted(dialogues[did], key=lambda x: x[0])
        edus = []
        cum = 0.0
        for _uid, r in rows:
            dur = max(0.0, _sec(r.get("EndTime")) - _sec(r.get("StartTime")))
            edus.append({
                "speaker": (r.get("Speaker") or "").strip(),
                "text": (r.get("Utterance") or "").strip(),
                "clip": [round(cum, 3), round(cum + dur, 3)],
                # per-EDU media segment (this utterance's own MELD clip). get_dialogue
                # promotes it to a served URL if the file exists; .mp4 (video) is
                # preferred over .flac (audio) when both are present.
                "clip_src": f"clips/{split}/dia{did}_utt{_uid}",
            })
            cum += dur
        rid = f"dia{did}"
        records.append({
            "id": rid,
            "edus": edus,
            # gold is empty (MELD is emotion); parser-predicted arcs overlay as a demo
            "relations": parsed.get(rid, []),
            "media": _media_for(did),
        })
    return records
