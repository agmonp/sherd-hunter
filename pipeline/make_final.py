"""
make_final.py — ONE merged video: animated intro -> LIVE dashboard walkthrough -> reflection ->
credits. Fixes: (1) opening is animated (curve draws, cards slide, band count ticks up) so the
first seconds aren't static; (2) the real dashboard footage sits before the credits.
Reuses make_video.py helpers + the recorded walkthrough (sherdhunter_walkthrough.mp4).
Out: sherdhunter_full.mp4. Run: python pipeline/make_final.py
"""
import os, sys, math, wave, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import make_video as mv
import make_walkthrough as mw
W, H, FPS = mv.W, mv.H, mv.FPS
WT = os.path.join(ROOT, "sherdhunter_walkthrough.mp4")


def sc_title():
    icon = Image.open(os.path.join(mv.VIEWER, "favicon.png")).convert("RGBA")
    def fr(p):
        im, d = mv.base()
        sc = int(110 + 46*min(p*2.2, 1)); ic = icon.resize((sc, sc)); im.paste(ic, (W//2-sc//2, 168), ic)
        ty = 372 - int(22*(1-min(p*1.6, 1)))
        t = "SherdHunter"; d.text(((W-d.textlength(t, font=mv.F_TITLE))/2, ty), t, font=mv.F_TITLE, fill=mv.TXT)
        if p > 0.42:
            s = "Finding ancient pottery scatters from space"
            d.text(((W-d.textlength(s, font=mv.F_SUB))/2, 452), s, font=mv.F_SUB, fill=mv.TEAL)
        return im
    return fr


def sc_fingerprint():
    def fr(p):
        im, d = mv.base(); mv.caption_chip(d, "The firing fingerprint")
        cx, cy, w = W//2, 330, 760; n = max(2, int(w*min(p*1.35, 1))); pts = []
        for i in range(n):
            t = i/w; dip = 60*math.exp(-((t-0.62)**2)/(2*0.006)) + 26*math.exp(-((t-0.42)**2)/(2*0.01))
            pts.append((cx-w//2+i, cy + 8*math.sin(t*2.2) + dip))
        if len(pts) > 1:
            d.line(pts, fill=mv.TEAL, width=5, joint="curve")
        if p > 0.72:
            xx = cx-w//2+int(w*0.62); d.line([(xx, cy-70), (xx, cy+95)], fill=(68, 85, 102), width=1)
            d.text((xx+8, cy-78), "2200 nm  (clay / firing)", font=mv.F_SM, fill=mv.MUT)
        return im
    return fr


def sc_data():
    def fr(p):
        im, d = mv.base(); mv.caption_chip(d, "Free hyperspectral satellites")
        for i, (nm, sub) in enumerate([("EMIT", "NASA · 60 m"), ("EnMAP", "DLR · 30 m")]):
            slide = int((1-min(p*2.2, 1)) * (340 if i == 0 else -340)); x = 360 + i*560 + slide
            d.rounded_rectangle([x-150, 230, x+150, 400], radius=16, fill=(20, 30, 40), outline=mv.TEAL, width=2)
            d.text((x-d.textlength(nm, font=mv.F_H)/2, 270), nm, font=mv.F_H, fill=mv.TXT)
            d.text((x-d.textlength(sub, font=mv.F_SUB)/2, 330), sub, font=mv.F_SUB, fill=mv.MUT)
        cnt = int(224*min(p*1.7, 1)); big = f"{cnt} colours · every 30-metre pixel"
        d.text(((W-d.textlength(big, font=mv.F_CAP))/2, 470), big, font=mv.F_CAP, fill=mv.GOLD)
        return im
    return fr


# (animated fn, narration); narration reused from make_video.SCENES for the shared beats
SEQ = [
    (sc_title(), mv.SCENES[0][2]),
    (sc_fingerprint(), mv.SCENES[1][2]),
    (sc_data(), mv.SCENES[2][2]),
    ("CLIP", mw.NARR),                          # live dashboard walkthrough
    (mv.sc_limits(), mv.SCENES[6][2]),
    (mv.sc_vision(), mv.SCENES[7][2]),
    (mv.sc_credits(), mv.SCENES[8][2]),
]


def pad720(frame):
    im = Image.fromarray(frame).convert("RGB")
    if im.width != W:
        im = im.resize((W, int(im.height*W/im.width)))
    cv = Image.new("RGB", (W, H), (8, 12, 16)); cv.paste(im, (0, (H-im.height)//2)); return cv


def main():
    import imageio
    tmp = tempfile.mkdtemp(); vid = os.path.join(tmp, "v.mp4")
    writer = imageio.get_writer(vid, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    audio = []; params = None
    for i, (fn, narr) in enumerate(SEQ):
        wav = os.path.join(tmp, f"s{i}.wav"); dur, params = mv.tts(narr, wav)
        if fn == "CLIP":
            rd = imageio.get_reader(WT); nf = 0
            for frame in rd:
                a = np.asarray(pad720(frame)); writer.append_data(a); writer.append_data(a); nf += 2
            rd.close()
            total = nf / FPS
            print(f"  clip: {nf} frames ({total:.1f}s), narration {dur:.1f}s")
        else:
            total = dur + 0.3; nfr = int(total*FPS)
            print(f"  scene {i}: {dur:.1f}s -> {nfr} frames")
            for k in range(nfr):
                p = k/max(nfr-1, 1)
                im = fn(min(p*1.5, 1.0)).convert("RGB"); d = ImageDraw.Draw(im); mv.draw_subtitle(d, narr)
                fade = 1.0
                if k < 6: fade = k/6
                elif k > nfr-7: fade = (nfr-1-k)/6
                if fade < 1.0:
                    im = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), im, max(fade, 0))
                writer.append_data(np.asarray(im))
        audio.append((wav, total))
    writer.close()

    aud = os.path.join(tmp, "a.wav")
    with wave.open(aud, "w") as out:
        out.setparams(params)
        for wavp, total in audio:
            with wave.open(wavp) as w:
                out.writeframes(w.readframes(w.getnframes())); used = w.getnframes()/w.getframerate()
            sil = int(max(0, total-used)*params.framerate)
            out.writeframes(b"\x00" * sil * params.sampwidth * params.nchannels)

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe(); out_mp4 = os.path.join(ROOT, "sherdhunter_full.mp4")
    subprocess.run([ff, "-y", "-i", vid, "-i", aud, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", out_mp4],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("WROTE", out_mp4, round(os.path.getsize(out_mp4)/1e6, 1), "MB")


if __name__ == "__main__":
    main()
