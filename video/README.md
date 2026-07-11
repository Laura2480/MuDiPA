# MuDiPA demo video — production pipeline

Target: EMNLP 2026 demo video — **≤ 2.5 min**, screencast + narration, MPEG4.
Focused storyboard (per Nils): Reasoning engine + Multimodal breathe; Export cut.

## Pieces
- `gen_audio.py` — TTS (edge-tts, no API key) → one MP3 per scene in `audio/`.
- `build_audio.sh` logic already run → `audio/narration_full.m4a` (**2:02**, timings baked in).
- `record_demo.py` — Playwright drives the app, records the viewport, paces each
  scene to its narration length → `raw/*.webm`.
- `build_video.py` — muxes the newest `raw/` recording with the narration → `mudipa_demo.mp4` (1080p).

## Per-scene timing (video = padded audio)
s1 intro 19s · s2 link 18s · s3 relation 16s · s4 **reasoning 30s** · s5 **multimodal 26s** · s6 close 12s → **~2:01**.

## Path A — fully automated (Playwright)
1. Start servers: app :5050 + DDPE :8092.
2. **Calibrate once:** open the app at 1280×720, screenshot, set the pixel `COORDS`
   in `record_demo.py` (edu_node, lightbulb, accept_chip, arc_midpoint, reasoning_star)
   and `MULTIMODAL_DATASET` / `DEMO_DIALOGUE_INDEX`.
3. `python video/record_demo.py`  → `video/raw/*.webm`
4. `python video/build_video.py`   → `video/mudipa_demo.mp4`

## Path B — OBS-guided (robust fallback)
Use if the canvas interactions are still changing. The audio is the metronome.
1. OBS: Window Capture of the browser, canvas 1920×1080, MP4/MKV, 30 fps.
2. Play `audio/narration_full.m4a`; perform each scene's actions for its allotted
   seconds (table above). Save to `video/raw/screen.mkv`.
3. `python video/build_video.py` → muxes with the same narration track.

## Re-timing / re-voicing
Edit the scene texts in `gen_audio.py`, rerun it, rebuild `narration_full.m4a`
(pad+concat), and update `SCENE_SECS` in `record_demo.py` to the new durations.
Swap voice via `VOICE` (e.g. `en-GB-SoniaNeural`) or trim time via `RATE="+8%"`.

## Title + closing cards (`mudipa_demo_full.mp4`)
`intro.html` (logo, objective, feature list, "now the demo") and `outro.html`
(logo + "thank you") are rendered at 1920×1080 and concatenated around the demo:
intro (8 s) + `mudipa_demo.mp4` + outro (4 s) → **`mudipa_demo_full.mp4`** (2:28, ≤ 2.5 min,
MPEG4). This is the screencast referenced in the top-level README. Rebuild: screenshot the
two HTML pages, then `ffmpeg` concat (silent AAC on the cards; short fades).

## Closing link overlay
`record_demo.py` opens the export dropdown on the last scene; add the repo/live-demo
URL as a lightweight text overlay in the final edit (or via an ffmpeg `drawtext`).
