"""
Generate the demo voice-over from the storyboard narration, one MP3 per scene,
using edge-tts (free Microsoft neural voices, no API key). Per-scene files let the
Playwright recorder wait exactly each scene's audio duration -> perfect A/V sync.

Run:  .venv/Scripts/python.exe video/gen_audio.py
"""
import asyncio
import edge_tts

VOICE = "en-US-AriaNeural"   # clear, neutral US voice; swap for en-GB-SoniaNeural / en-US-GuyNeural
RATE = "+0%"                 # speed; use e.g. "+8%" to shave time if needed
OUT = "video/audio"

# (id, text) — the FOCUSED storyboard (per Nils: fewer features, Reasoning + Multimodal breathe)
SCENES = [
    ("s1_intro",
     "MuDiPA is a human-in-the-loop platform for annotating discourse structure in "
     "multi-party dialogue. It lays each utterance on a pan-and-zoom canvas and drafts a "
     "discourse graph, with labelled relations between the utterances."),
    ("s2_link",
     "To build the graph, the annotator selects an utterance and the suggestion engine "
     "proposes its most likely attachment, with a confidence score. A single click accepts "
     "it."),
    ("s3_relation",
     "The link is then labelled: the codebook defines each SDRT relation, and the parser "
     "ranks the likely labels for the annotator to choose."),
    ("s4_reasoning",
     "The reasoning engine then vets that choice against the relation's own definition. It is "
     "given a single relation and asked only whether it holds. Here it finds that Elaboration "
     "does not hold, since yep plenty answers the question rather than adding any detail. Across "
     "our study, it caught such mislabellings correctly in over four out of five cases."),
    ("s_sub",
     "MuDiPA also groups the dialogue into sub-dialogues, the separate conversational threads "
     "running through it, and can explain each one."),
    ("s5_multimodal",
     "MuDiPA is not limited to text. For multimodal corpora, video, audio, and text stay "
     "synchronized: selecting an utterance seeks the player to that moment, so every unit is "
     "annotated in its original context."),
    ("s6_close",
     "MuDiPA supports multiple corpora and pluggable parser and language-model back-ends. "
     "Try the live demo and the code at the link on screen."),
]


async def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    for sid, text in SCENES:
        path = f"{OUT}/{sid}.mp3"
        await edge_tts.Communicate(text, VOICE, rate=RATE).save(path)
        print(f"  wrote {path}")
    # also write the full script for reference / re-timing
    with open(f"{OUT}/narration.txt", "w", encoding="utf-8") as f:
        for sid, text in SCENES:
            f.write(f"[{sid}] {text}\n\n")
    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
