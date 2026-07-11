"""
Final assembly: mux a screen recording with the narration track into an
EMNLP-compliant MPEG4 (H.264 + AAC), scaled/padded to 1080p.

Works with ANY screen recording — the Playwright WEBM (video/raw/*.webm) or an
OBS capture you drop in video/raw/. Picks the newest file in video/raw/.

Run:  .venv/Scripts/python.exe video/build_video.py
Out:  video/mudipa_demo.mp4
"""
import glob
import os
import subprocess


def find_ffmpeg():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    hits = glob.glob(os.path.expanduser(
        "~/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
    if hits:
        return hits[0]
    raise SystemExit("ffmpeg not found — install it (winget install Gyan.FFmpeg)")


def newest_raw():
    vids = sorted(glob.glob("video/raw/*.webm") + glob.glob("video/raw/*.mkv")
                  + glob.glob("video/raw/*.mp4"), key=os.path.getmtime)
    if not vids:
        raise SystemExit("no screen recording in video/raw/ — run record_demo.py or drop an OBS file there")
    return vids[-1]


def main():
    ff = find_ffmpeg()
    screen = newest_raw()
    audio = "video/audio/narration_full.m4a"
    out = "video/mudipa_demo.mp4"
    trim = 0.0
    try:
        trim = float(open("video/_trim.txt").read().strip())   # seconds of setup/login to cut
    except Exception:
        pass
    print(f"video: {screen}\naudio: {audio}\ntrim: {trim:.2f}s\n-> {out}")
    # scale to fit 1920x1080 keeping AR, pad to 1080p; H.264 + AAC; end at audio length.
    vf = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30")
    cmd = [ff, "-y", "-ss", f"{trim:.2f}", "-i", screen, "-i", audio,
           "-map", "0:v:0", "-map", "1:a:0",
           "-vf", vf,
           "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k",
           "-shortest", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print("DONE ->", out)


if __name__ == "__main__":
    main()
