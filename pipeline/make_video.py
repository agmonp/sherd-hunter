"""
make_video.py — produce the SherdHunter explainer (≈2 min, English TTS narration, burned-in
captions/subtitles). Composed entirely from assets + PIL graphics (no screen-recording).
Tooling already present: PIL, imageio + imageio_ffmpeg (bundled ffmpeg), Windows SAPI TTS.

Out: sherdhunter_intro.mp4 (project root). Run: python pipeline/make_video.py
"""
import os, sys, math, wave, subprocess, tempfile, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
W, H, FPS = 1280, 720, 30
BG = (14, 21, 28); TEAL = (52, 200, 180); ORANGE = (255, 122, 60); RED = (230, 50, 50)
TXT = (232, 238, 244); MUT = (141, 162, 181); GOLD = (245, 197, 66)


def font(sz, bold=False):
    for p in ([r"C:\Windows\Fonts\segoeuib.ttf"] if bold else [r"C:\Windows\Fonts\segoeui.ttf"]) + \
             [r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

F_TITLE, F_H, F_CAP, F_SUB, F_SM = font(64, True), font(34, True), font(30, True), font(26), font(20)


def tts(text, wav):
    txt = wav + ".txt"; open(txt, "w", encoding="utf-8").write(text)
    ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "try{$s.SelectVoice('Microsoft David Desktop')}catch{}; $s.Rate=1; "          # brisker pace
          f"$s.SetOutputToWaveFile('{wav}'); $s.Speak([IO.File]::ReadAllText('{txt}')); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(wav) as w:
        return w.getnframes() / w.getframerate(), w.getparams()


def draw_subtitle(d, text, y=648):
    lines = textwrap.wrap(text, 74)
    d.rectangle([0, y-12, W, H], fill=(8, 12, 16))
    for i, ln in enumerate(lines[:2]):
        tw = d.textlength(ln, font=F_SUB)
        d.text(((W-tw)/2, y + i*30), ln, font=F_SUB, fill=TXT)


def caption_chip(d, text):
    tw = d.textlength(text, font=F_CAP)
    d.rounded_rectangle([(W-tw)/2-18, 70, (W+tw)/2+18, 122], radius=14, fill=(20, 30, 40), outline=TEAL, width=2)
    d.text(((W-tw)/2, 80), text, font=F_CAP, fill=TEAL)


def base():
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    for y in range(H):                                   # subtle vertical gradient
        d.line([(0, y), (W, y)], fill=(14, 21+int(6*y/H), 28+int(10*y/H)))
    return im, d


def fit(asset, maxw, maxh):
    im = Image.open(os.path.join(VIEWER, asset)).convert("RGBA")
    s = min(maxw/im.width, maxh/im.height); return im.resize((int(im.width*s), int(im.height*s)))


def spectral_graphic(d, cx, cy, w):
    pts = []
    for i in range(w):
        t = i/w; dip = 60*math.exp(-((t-0.62)**2)/(2*0.006)) + 26*math.exp(-((t-0.42)**2)/(2*0.01))
        pts.append((cx-w//2+i, cy + 8*math.sin(t*2.2) + dip))
    d.line(pts, fill=TEAL, width=5, joint="curve")
    xx = cx-w//2+int(w*0.62)
    d.line([(xx, cy-70), (xx, cy+95)], fill=(68, 85, 102), width=1)
    d.text((xx+8, cy-78), "2200 nm  (clay / firing)", font=F_SM, fill=MUT)


# ---------------- scene frame builders (return f(progress)->RGB Image) ----------------
def sc_title():
    icon = Image.open(os.path.join(VIEWER, "favicon.png")).convert("RGBA").resize((150, 150))
    def fr(p):
        im, d = base(); im.paste(icon, (W//2-75, 188), icon)
        t = "SherdHunter"; d.text(((W-d.textlength(t, font=F_TITLE))/2, 360), t, font=F_TITLE, fill=TXT)
        s = "Finding ancient pottery scatters from space"
        d.text(((W-d.textlength(s, font=F_SUB))/2, 440), s, font=F_SUB, fill=TEAL)
        return im
    return fr


def sc_fingerprint():
    def fr(p):
        im, d = base(); caption_chip(d, "The firing fingerprint")
        spectral_graphic(d, W//2, 330, 760)
        return im
    return fr


def sc_data():
    def fr(p):
        im, d = base(); caption_chip(d, "Free hyperspectral satellites")
        for i, (nm, sub) in enumerate([("EMIT", "NASA · 60 m"), ("EnMAP", "DLR · 30 m")]):
            x = 360 + i*560
            d.rounded_rectangle([x-150, 230, x+150, 400], radius=16, fill=(20, 30, 40), outline=TEAL, width=2)
            d.text((x-d.textlength(nm, font=F_H)/2, 270), nm, font=F_H, fill=TXT)
            d.text((x-d.textlength(sub, font=F_SUB)/2, 330), sub, font=F_SUB, fill=MUT)
        big = "224 colours · every 30-metre pixel"
        d.text(((W-d.textlength(big, font=F_CAP))/2, 470), big, font=F_CAP, fill=GOLD)
        return im
    return fr


def sc_asset(asset, chip, maxh=470, kb=0.06):
    art = fit(asset, 1120, maxh)
    def fr(p):
        im, d = base(); caption_chip(d, chip)
        z = 1 + kb*p; a2 = art.resize((int(art.width*z), int(art.height*z)))
        im.paste(a2, (W//2 - a2.width//2, 150 - int(kb*p*40)), a2)
        return im
    return fr


def sc_asset_overlay(asset, chip, label):
    art = fit(asset, 1120, 470)
    def fr(p):
        im, d = base(); caption_chip(d, chip)
        im.paste(art, (W//2-art.width//2, 150), art)
        if p > 0.35:                                     # reveal the "masked" stamp
            d.rounded_rectangle([W//2-220, 540, W//2+220, 600], radius=12, fill=(40, 20, 24), outline=RED, width=3)
            d.text((W//2-d.textlength(label, font=F_CAP)/2, 552), label, font=F_CAP, fill=(255, 150, 150))
        return im
    return fr


def sc_limits():
    def fr(p):
        im, d = base(); caption_chip(d, "A filter, not a map")
        d.rectangle([300, 250, 540, 490], outline=TEAL, width=4)
        d.text((300, 500), "30 m pixel", font=F_SUB, fill=MUT)
        d.ellipse([760, 360, 776, 376], fill=ORANGE)
        d.text((720, 500), "a sherd (cm)", font=F_SUB, fill=MUT)
        d.text((560, 350), "→", font=F_TITLE, fill=MUT)
        return im
    return fr


def sc_vision():
    def fr(p):
        im, d = base(); caption_chip(d, "Scalable · from open data")
        rng = np.random.default_rng(3)
        for gx in range(8):
            for gy in range(4):
                x, y = 280+gx*90, 250+gy*90
                on = (gx*4+gy) < int(p*32)
                d.rounded_rectangle([x, y, x+70, y+70], radius=8,
                                    fill=(30, 40, 50) if not on else (60, 30, 26), outline=(40, 54, 66))
                if on and rng.random() < 0.5:
                    d.ellipse([x+28, y+28, x+42, y+42], fill=ORANGE)
        return im
    return fr


def sc_credits():
    lines = ["Built with open data from", "NASA  ·  DLR (EnMAP)  ·  Geological Survey of Israel",
             "OpenStreetMap  ·  USGS spectral library"]
    def fr(p):
        im, d = base()
        for i, ln in enumerate(lines):
            f = F_H if i == 0 else F_SUB
            d.text(((W-d.textlength(ln, font=f))/2, 250+i*70), ln, font=f, fill=TXT if i == 0 else MUT)
        t = "SherdHunter"; d.text(((W-d.textlength(t, font=F_CAP))/2, 520), t, font=F_CAP, fill=TEAL)
        return im
    return fr


SCENES = [
    (sc_title(), None,
     "SherdHunter. Finding ancient pottery scatters from space."),
    (sc_fingerprint(), None,
     "Ancient settlement mounds, called tells, are built from thousands of years of human life — "
     "ash, lime, and broken pottery. Burning clay changes how it reflects light, near twenty-two "
     "hundred nanometres. That is a fingerprint you can read from orbit."),
    (sc_data(), None,
     "We use free hyperspectral satellites — NASA's EMIT and Germany's EnMAP. Instead of three "
     "colours, they record two hundred and twenty-four, for every thirty-metre patch of desert."),
    (sc_asset("detection.png", "The detection map"), None,
     "This is the result over Tel Arad and the Dead Sea. Red means a strong anomaly. On top we "
     "overlay what is already known — sites, geology, and modern features — so each hot spot can "
     "be explained, or flagged as new."),
    (sc_asset("verify.png", "Does it actually work?", 430), None,
     "Known sites we never trained on stand out from random ground with a score of zero point "
     "seven eight. The chance of that being luck is below one in twenty thousand. And a blind test "
     "on Tel Malhata, held out completely, landed near the top."),
    (sc_asset_overlay("detection.png", "Chasing the false alarms", "Hatrurim & Dead Sea — masked"), None,
     "Real science means chasing the false alarms. The brightest spots were not ruins — they were "
     "a natural formation that bakes its own carbonate, and the salt flats of the Dead Sea. We "
     "mapped and removed them, leaving a short, clean list of candidates."),
    (sc_limits(), None,
     "To be honest — at thirty metres we do not see pottery. We see the faint chemical shadow of a "
     "human-altered surface. Confirming it still needs drones, and boots on the ground."),
    (sc_vision(), None,
     "But it is a filter that points us at the most promising places — reproducibly, from open "
     "data, across the whole desert."),
    (sc_credits(), None,
     "Built with open data from NASA, the German Aerospace Center, the Geological Survey of "
     "Israel, OpenStreetMap, and the U S G S."),
]


def main():
    tmp = tempfile.mkdtemp()
    vid = os.path.join(tmp, "v.mp4")
    import imageio
    writer = imageio.get_writer(vid, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    audio_parts = []; params = None
    for i, (fr, _chip, narr) in enumerate(SCENES):
        wav = os.path.join(tmp, f"s{i}.wav")
        dur, params = tts(narr, wav)
        total = dur + 0.3                                # small tail
        nfr = int(total * FPS)
        print(f"  scene {i}: {dur:.1f}s narration -> {nfr} frames")
        for k in range(nfr):
            p = k / max(nfr-1, 1)
            im = fr(min(p*1.6, 1.0)).convert("RGB")
            d = ImageDraw.Draw(im); draw_subtitle(d, narr)
            fade = 1.0
            if k < 6: fade = k/6
            elif k > nfr-7: fade = (nfr-1-k)/6
            if fade < 1.0:
                im = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), im, max(fade, 0))
            writer.append_data(np.asarray(im))
        audio_parts.append((wav, total))
    writer.close()

    # concatenate narration WAVs + silence tails into one track
    aud = os.path.join(tmp, "a.wav")
    with wave.open(aud, "w") as out:
        out.setparams(params)
        for wavp, total in audio_parts:
            with wave.open(wavp) as w:
                out.writeframes(w.readframes(w.getnframes()))
            sil = int((total - w.getnframes()/w.getframerate()) * params.framerate)
            out.writeframes(b"\x00" * sil * params.sampwidth * params.nchannels)

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out_mp4 = os.path.join(ROOT, "sherdhunter_intro.mp4")
    subprocess.run([ff, "-y", "-i", vid, "-i", aud, "-c:v", "copy", "-c:a", "aac", "-shortest", out_mp4],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("WROTE", out_mp4, round(os.path.getsize(out_mp4)/1e6, 1), "MB")


if __name__ == "__main__":
    main()
