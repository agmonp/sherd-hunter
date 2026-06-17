"""
export_for_viewer.py — turn the real extracted EMIT spectra (data/*.npy) into the
JSON the browser viewer reads. Also emits the known-sites layer and a CLEARLY
LABELLED *synthetic* demo heatmap so the UI is populated before a full-scene ACE
run on the real cube exists.

Run:  python pipeline/export_for_viewer.py
Outputs (viewer/):  spectra.json, sites.geojson, detection_demo.json
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "viewer")
os.makedirs(OUT, exist_ok=True)

# ---- real data --------------------------------------------------------------
wl      = np.load(os.path.join(DATA, "wavelengths.npy")).astype(float)      # 285
tell    = np.load(os.path.join(DATA, "tell_spectrum.npy")).astype(float)    # 285
bg      = np.load(os.path.join(DATA, "bg_spectrum.npy")).astype(float)      # 285
cr_wl   = np.load(os.path.join(DATA, "cr_wl.npy")).astype(float)            # 44
cr_tell = np.load(os.path.join(DATA, "cr_ontell.npy")).astype(float)        # 44
cr_bg   = np.load(os.path.join(DATA, "cr_background.npy")).astype(float)    # 44

# EMIT deep water-vapor bands: blank them so plotted lines break instead of diving to noise.
WV = [(1340, 1465), (1790, 1960)]
FILL = -0.01
def clean(spec):
    s = spec.copy()
    s[s <= FILL + 1e-6] = np.nan
    for a, b in WV:
        s[(wl >= a) & (wl <= b)] = np.nan
    return s

def depth_at(cw, cs, center, halfwin=15.0):
    """absorption depth (1 - continuum-removed reflectance) near `center` nm."""
    m = (cw >= center - halfwin) & (cw <= center + halfwin)
    if not m.any():
        return None
    return float(1.0 - np.nanmin(cs[m]))

def brightness(spec):
    m = ((wl >= 500) & (wl <= 1300)) & np.isfinite(spec)
    return float(np.nanmean(spec[m]))

tell_c, bg_c = clean(tell), clean(bg)

def series(x, y):
    return [[round(float(a), 2), (None if not np.isfinite(b) else round(float(b), 5))]
            for a, b in zip(x, y)]

spectra = {
    "meta": {
        "sensor": "NASA EMIT L2A reflectance",
        "granule": "EMIT_L2A_RFL_001_20250616T104810_2516707_015",
        "target": "Tel Arad, Israel",
        "n_onTell": 1, "n_background": 1,
        "note": "Single on-tell pixel vs single background pixel. Anecdote, not statistics.",
        "diagnostic_windows": {"AlOH_clay": [2160, 2230], "carbonate": [2320, 2360]},
    },
    "full": {              # full 0.38-2.49 um reflectance
        "wl": [round(float(w), 2) for w in wl],
        "onTell": series(wl, tell_c),
        "background": series(wl, bg_c),
    },
    "continuum": {         # continuum-removed Al-OH/carbonate window (the diagnostic zoom)
        "wl": [round(float(w), 2) for w in cr_wl],
        "onTell": [round(float(v), 5) for v in cr_tell],
        "background": [round(float(v), 5) for v in cr_bg],
    },
    "features": {
        "carbonate_2345_depth": {"onTell": depth_at(cr_wl, cr_tell, 2345),
                                 "background": depth_at(cr_wl, cr_bg, 2345)},
        "alOH_2200_depth":      {"onTell": depth_at(cr_wl, cr_tell, 2200),
                                 "background": depth_at(cr_wl, cr_bg, 2200)},
        "brightness_vis_nir":   {"onTell": brightness(tell_c),
                                 "background": brightness(bg_c)},
    },
}
json.dump(spectra, open(os.path.join(OUT, "spectra.json"), "w"), indent=1)

# ---- sites + sample pixels --------------------------------------------------
TEL_ARAD = (35.1247, 31.2803)
BG_PT    = (35.1564, 31.2803)
sites = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"name": "Tel Arad", "kind": "target",
            "note": "EB II lower city; dense surface sherd scatter; IAA ground truth."},
         "geometry": {"type": "Point", "coordinates": list(TEL_ARAD)}},
        {"type": "Feature", "properties": {"name": "On-tell sample pixel", "kind": "sample_hot",
            "note": "EMIT pixel r30/c522, 31 m from target. Carbonate depth 0.0675."},
         "geometry": {"type": "Point", "coordinates": list(TEL_ARAD)}},
        {"type": "Feature", "properties": {"name": "Background sample pixel", "kind": "sample_cold",
            "note": "Open desert ~3 km E. Carbonate depth 0.0457."},
         "geometry": {"type": "Point", "coordinates": list(BG_PT)}},
        {"type": "Feature", "properties": {"name": "Tel Beer-Sheva", "kind": "site",
            "note": "UNESCO Iron Age tell, ~28 km W. Good second validation target."},
         "geometry": {"type": "Point", "coordinates": [34.8413, 31.2447]}},
        {"type": "Feature", "properties": {"name": "Khirbet Qumran", "kind": "site",
            "note": "From legacy argis sites list."},
         "geometry": {"type": "Point", "coordinates": [35.458, 31.741]}},
    ],
}
json.dump(sites, open(os.path.join(OUT, "sites.geojson"), "w"), indent=1)

# ---- DEMO heatmap (SYNTHETIC — labelled) ------------------------------------
# Illustrates the intended product: an ACE detection-score surface. Peak sits on the
# tell (where our 1 real sample shows the anomaly); one off-site "candidate" decoy
# demonstrates the flag-anomalies-outside-known-sites use case. NOT a real detection.
rng = np.random.default_rng(42)
LON0, LAT0, LON1, LAT1 = 35.05, 31.23, 35.20, 31.33
nx, ny = 120, 90
xs = np.linspace(LON0, LON1, nx)
ys = np.linspace(LAT0, LAT1, ny)
def blob(LON, LAT, cx, cy, amp, sx, sy):
    return amp * np.exp(-(((LON - cx) / sx) ** 2 + ((LAT - cy) / sy) ** 2))
LON, LAT = np.meshgrid(xs, ys)
field = (blob(LON, LAT, 35.1247, 31.2803, 1.00, 0.010, 0.008)    # Tel Arad (real anomaly site)
       + blob(LON, LAT, 35.092, 31.262, 0.55, 0.007, 0.006)      # off-site candidate decoy
       + blob(LON, LAT, 35.168, 31.300, 0.30, 0.006, 0.005))     # carbonate outcrop false-alarm
field += rng.normal(0, 0.04, field.shape).clip(0)
field = (field / field.max()).clip(0, 1)
pts = [[round(float(ys[j]), 4), round(float(xs[i]), 4), round(float(field[j, i]), 3)]
       for j in range(ny) for i in range(nx) if field[j, i] > 0.45]   # markers: strong hits only
demo = {"label": "SYNTHETIC DEMO — not a real detection (full-scene ACE pending cube)",
        "bbox": [LON0, LAT0, LON1, LAT1], "max": 1.0, "points": pts}
json.dump(demo, open(os.path.join(OUT, "detection_demo.json"), "w"))

# Render the field to a smooth RGBA PNG the map overlays directly (turbo-ish colormap).
def colormap(v):
    stops = [(0.0,(40,30,90)),(0.25,(30,120,200)),(0.5,(40,200,140)),
             (0.7,(230,220,40)),(0.85,(240,140,30)),(1.0,(220,30,30))]
    for k in range(len(stops)-1):
        v0,c0 = stops[k]; v1,c1 = stops[k+1]
        if v <= v1:
            t = 0 if v1==v0 else (v-v0)/(v1-v0)
            return tuple(int(c0[i]+(c1[i]-c0[i])*t) for i in range(3))
    return stops[-1][1]
try:
    from PIL import Image
    UP = 6  # upsample for a smooth look
    big = np.kron(field, np.ones((UP, UP)))
    H, W = big.shape
    img = np.zeros((H, W, 4), dtype=np.uint8)
    for j in range(H):
        for i in range(W):
            v = big[j, i]
            r, g, b = colormap(v)
            a = 0 if v < 0.15 else int(min(1.0, (v-0.15)/0.5) * 200)
            img[j, i] = (r, g, b, a)
    Image.fromarray(np.flipud(img), "RGBA").save(os.path.join(OUT, "detection_demo.png"))
    print("wrote detection_demo.png", (W, H))
except Exception as e:
    print("PNG render skipped:", e)

# ---- bundle everything into data.js so index.html works via file:// (no fetch) ----
import base64
png_path = os.path.join(OUT, "detection_demo.png")
png_uri = ""
if os.path.exists(png_path):
    png_uri = "data:image/png;base64," + base64.b64encode(open(png_path, "rb").read()).decode()
bundle = {"spectra": spectra, "sites": sites, "detectionDemo": demo, "detectionPng": png_uri}
with open(os.path.join(OUT, "data.js"), "w", encoding="utf-8") as f:
    f.write("window.SHERD = " + json.dumps(bundle) + ";")

print("wrote spectra.json, sites.geojson, detection_demo.json, data.js")
print("carbonate 2345 depth  on-tell=%.4f  bg=%.4f"
      % (spectra["features"]["carbonate_2345_depth"]["onTell"],
         spectra["features"]["carbonate_2345_depth"]["background"]))
print("heatmap demo points:", len(pts))
