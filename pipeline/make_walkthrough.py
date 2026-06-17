"""make_walkthrough.py — turn the screen-recorded auto-demo (scenes/_demo_raw.mp4) into a
narrated live UI walkthrough: crop browser chrome + taskbar, trim the lead-in, add English
voiceover (the on-screen captions already label each step). Out: sherdhunter_walkthrough.mp4."""
import os, subprocess, wave
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")
FF = imageio_ffmpeg.get_ffmpeg_exe()
DN = subprocess.DEVNULL
RAW = os.path.join(SCENES, "_demo_raw.mp4")
CLEAN = os.path.join(SCENES, "_demo_clean.mp4")

NARR = ("This is the SherdHunter dashboard, running live over the Negev desert. "
        "The coloured layer is the firing screen — our strongest archaeological signal; red means a "
        "strong anomaly. We overlay what we already know. Every known archaeological site, as ground "
        "truth. The national geology map, including the Hatrurim and Lisan formations that the "
        "detector also lights up. And a mask over water, salt flats and built-up areas, where pottery "
        "cannot be — so the Dead Sea drops out. A second screen shows the broad carbonate halo, with "
        "adjustable opacity. In the Findings tab, every result is validated: known sites beat random "
        "ground with a score of zero point seven eight, confirmed by a permutation test and a spectral "
        "check on the real absorption lines. Open data, reproducible — from orbit to outcrop.")


def tts(text, wav, rate=1):
    txt = wav + ".txt"; open(txt, "w", encoding="utf-8").write(text)
    ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "try{$s.SelectVoice('Microsoft David Desktop')}catch{}; $s.Rate=%d;"
          "$s.SetOutputToWaveFile('%s'); $s.Speak([IO.File]::ReadAllText('%s')); $s.Dispose()" % (rate, wav, txt))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, stdout=DN, stderr=DN)
    with wave.open(wav) as w:
        return w.getnframes() / w.getframerate()


def duration(path):
    import imageio_ffmpeg, json
    # use ffmpeg to print duration
    p = subprocess.run([FF, "-i", path], stderr=subprocess.PIPE, stdout=DN, text=True)
    for ln in p.stderr.splitlines():
        if "Duration:" in ln:
            h, m, s = ln.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h)*3600 + int(m)*60 + float(s)
    return 0.0


def main():
    # base: crop chrome(top 112)+taskbar; trim 4.5s lead-in; scale to 1280 wide; fade in
    base = os.path.join(SCENES, "_demo_base.mp4")
    subprocess.run([FF, "-y", "-ss", "4.5", "-i", RAW,
                    "-vf", "crop=1920:920:0:112,scale=1280:-2,fade=t=in:st=0:d=0.6",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", base], check=True, stdout=DN, stderr=DN)
    bdur = duration(base)
    wav = os.path.join(SCENES, "_wt.wav")
    adur = tts(NARR, wav)
    pad = max(0.0, adur + 0.6 - bdur)              # hold last frame to cover the narration
    fo = max(0.5, bdur + pad - 0.8)
    print(f"base video {bdur:.1f}s, narration {adur:.1f}s, pad {pad:.1f}s")
    subprocess.run([FF, "-y", "-i", base, "-vf",
                    f"tpad=stop_mode=clone:stop_duration={pad:.2f},fade=t=out:st={fo:.2f}:d=0.8",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", CLEAN], check=True, stdout=DN, stderr=DN)
    out = os.path.join(ROOT, "sherdhunter_walkthrough.mp4")
    subprocess.run([FF, "-y", "-i", CLEAN, "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", out], check=True, stdout=DN, stderr=DN)
    print("WROTE", out, round(os.path.getsize(out)/1e6, 1), "MB")


if __name__ == "__main__":
    main()
