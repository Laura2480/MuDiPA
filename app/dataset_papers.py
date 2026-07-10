"""Dataset -> source-publication provenance for MuDiPA.

Connects every dataset to the paper that introduced it, reusing the MuDiPA paper's own
BibTeX keys (paper/references.bib) where they exist, so the app and the paper cite the
same works. Surfaced in the UI (dataset picker) and via `/api/config`, and handy for the
paper's dataset table.

`paper_for(dataset, cfg)` returns `{bibkey, short, title, venue, year, url}` (+ optional
`also` for a second reference), or None when a dataset has no source paper.
"""

# Keyed by BibTeX key (matches paper/references.bib for the shared ones).
_PAPERS = {
    "asher2016stac": dict(
        short="Asher et al. 2016", year=2016,
        title="Discourse Structure and Dialogue Acts in Multiparty Dialogue (STAC)",
        venue="LREC 2016", url="https://aclanthology.org/L16-1432/"),
    "li2020molweni": dict(
        short="Li et al. 2020", year=2020,
        title="Molweni: A Challenge Multiparty Dialogues-based MRC Dataset with Discourse Structure",
        venue="COLING 2020", url="https://aclanthology.org/2020.coling-main.238/"),
    "thompson2024discourse": dict(
        short="Thompson et al. 2024", year=2024,
        title="Discourse Structure for the Minecraft Corpus (MSDC)",
        venue="2024", url=""),
    "liu2026draddp": dict(
        short="Liu et al. 2026", year=2026,
        title="DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset",
        venue="Findings of ACL 2026", url="https://arxiv.org/abs/2606.00012"),
    "poria2019meld": dict(
        short="Poria et al. 2019", year=2019,
        title="MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation",
        venue="ACL 2019", url="https://arxiv.org/abs/1810.02508"),
    "moddp2024": dict(
        short="Gong et al. 2024", year=2024,
        title="MODDP: A Multi-modal Open-domain Chinese Dataset for Dialogue Discourse Parsing",
        venue="2024", url=""),
    "shukla2025comumdr": dict(
        short="Shukla et al. 2025", year=2025,
        title="CoMuMDR: Code-mixed Multi-modal Multi-domain Corpus for Discourse Parsing",
        venue="2025", url="https://arxiv.org/abs/2506.08504"),
    # DISRPT corpora share the shared-task reference (not yet in the paper's .bib).
    "disrpt2025": dict(
        short="Braud et al. 2025", year=2025,
        title="The DISRPT 2025 Shared Task on EDU Segmentation, Connective Detection, "
              "and Relation Classification",
        venue="DISRPT 2025 (ACL)", url="https://aclanthology.org/2025.disrpt-1.1/"),
}

# Native / explicitly-registered datasets -> bibkey.
_DATASET_KEY = {
    "stac": "asher2016stac",
    "stac_full": "asher2016stac",
    "eng.sdrt.stac": "asher2016stac",
    "molweni": "li2020molweni",
    "eng.sdrt.msdc": "thompson2024discourse",
    "draddp": "poria2019meld",     # actual data is MELD; the DraDDP-style task ref is `also`
    "moddp": "moddp2024",
}


def paper_for(dataset, cfg=None):
    key = _DATASET_KEY.get(dataset)
    if not key and (cfg or {}).get("loader") == "disrpt":
        key = "disrpt2025"          # every DISRPT corpus -> the shared task
    if not key or key not in _PAPERS:
        return None
    out = dict(_PAPERS[key])
    out["bibkey"] = key
    if dataset == "draddp":         # MELD data, DraDDP-style discourse task
        also = dict(_PAPERS["liu2026draddp"]); also["bibkey"] = "liu2026draddp"
        out["also"] = also
    return out
