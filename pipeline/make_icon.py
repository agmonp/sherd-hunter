"""make_icon.py — generate the SherdHunter app icon (spectral curve + detection hotspot)
-> viewer/favicon.png (browser) + viewer/favicon.ico (desktop shortcut). Pure PIL."""
import os, math
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# rounded-square background (app dark) + border
d.rounded_rectangle([6, 6, S-6, S-6], radius=46, fill=(14, 21, 28, 255), outline=(40, 54, 66, 255), width=4)

# spectral curve: gentle baseline with a Gaussian absorption dip (the 2200 nm feature), teal
pts = []
for x in range(28, S-28):
    t = (x - 28) / (S - 56)
    dip = 30 * math.exp(-((t - 0.55) ** 2) / (2 * 0.012))    # one shallow absorption dip
    y = 112 + 6 * math.sin(t * 2.2) + dip
    pts.append((x, y))
d.line(pts, fill=(52, 200, 180, 255), width=10, joint="curve")

# detection hotspot (radial orange->red glow) at the dip bottom
cx, cy = pts[int(len(pts) * 0.56)]
for r, col in [(46, (230, 50, 50, 40)), (34, (240, 120, 40, 70)),
               (22, (255, 140, 60, 140)), (13, (255, 170, 80, 255))]:
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)
# target ring
d.ellipse([cx-30, cy-30, cx+30, cy+30], outline=(245, 197, 66, 230), width=4)

img.save(os.path.join(VIEWER, "favicon.png"))
img.save(os.path.join(VIEWER, "favicon.ico"), sizes=[(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)])
print("wrote viewer/favicon.png + viewer/favicon.ico")
