"""
verify.py — two independent checks that the EnMAP composite result is real, not an artifact.

(1) SPECTRAL SANITY: pull the actual EnMAP reflectance at a blind-test hit (Tel Malhata), a
    Hatrurim hotspot, Tel Arad, and a matched-terrain background pixel. Measure the diagnostic
    continuum-removed depths (carbonate 2345 nm, Al-OH 2200 nm). If "hot" points genuinely show
    the carbonate absorption the index claims, the score is grounded in real spectroscopy.

(2) PERMUTATION NULL TEST: the reported known-site AUC (0.80) — is it beyond chance? Shuffle the
    site/background labels M times, recompute AUC each time (fast rank-sum), and locate the
    observed AUC in that null distribution (z-score, empirical p). Rules out coincidence.

In : scenes/enmap_DT0000069168_SPECTRAL.tiff, scenes/score_composite_enmap.npz,
     viewer/known_sites.geojson
Out: viewer/verify.png, viewer/verify.json, console
Run: python pipeline/verify.py
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
SCENES = os.path.join(ROOT, "scenes")
TIF = os.path.join(SCENES, "enmap_DT0000069168_SPECTRAL.tiff")
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from composite_enmap import enmap_wavelengths

# name, lon, lat, colour, kind
POINTS = [
    ("Tel Malhata (blind hit)", 35.0417, 31.2306, "#ff7a3c", "site"),
    ("Hatrurim hotspot",        35.2936, 31.1598, "#ff4d4d", "geology"),
    ("Tel Arad (ref)",          35.1247, 31.2803, "#f5c542", "site"),
    ("background plateau",       35.1564, 31.2803, "#34c8b4", "background"),
]


def cr_depth(wl, sp, w0, w1):
    m = (wl >= w0) & (wl <= w1)
    w, S = wl[m], sp[m]
    if S.size < 3 or not np.isfinite(S).all():
        return np.nan
    a, b = S[0], S[-1]
    t = (w - w[0]) / (w[-1] - w[0])
    cont = a + (b - a) * t
    return float(1.0 - np.nanmin(S / np.clip(cont, 1e-6, None)))


def auc_ranksum(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    n1, n2 = len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2), ranks, n1, n2


def main():
    import rasterio
    from rasterio.warp import transform as warp_xy
    from rasterio.transform import rowcol
    t0 = time.time()
    ds = rasterio.open(TIF); nod = ds.nodata; wl = enmap_wavelengths()
    H, W = ds.height, ds.width

    # (1) spectra at the points (3x3 mean for stability)
    specs = []
    for name, lon, lat, col, kind in POINTS:
        x, y = warp_xy("EPSG:4326", ds.crs, [lon], [lat])
        r, c = rowcol(ds.transform, x, y)
        r, c = int(np.atleast_1d(r)[0]), int(np.atleast_1d(c)[0])
        r0, c0 = max(0, r-1), max(0, c-1)
        win = ds.read(window=((r0, min(H, r+2)), (c0, min(W, c+2)))).astype(np.float32)
        win[win == nod] = np.nan; win[win <= 0] = np.nan
        sp = np.nanmean(win.reshape(win.shape[0], -1), axis=1) * 1e-4
        specs.append({"name": name, "color": col, "kind": kind, "sp": sp,
                      "carb2345": cr_depth(wl, sp, 2250, 2400),   # WIDE window (matches composite)
                      "alOH2200": cr_depth(wl, sp, 2120, 2245)})
    print("(1) SPECTRAL SANITY — continuum-removed depths:")
    for s in specs:
        print(f"    {s['name']:26s} carbonate2345={s['carb2345']:.4f}  alOH2200={s['alOH2200']:.4f}  ({s['kind']})")

    # (2) permutation null on the known-site AUC (anomaly grid)
    d = np.load(os.path.join(SCENES, "score_composite_enmap.npz"))
    sc = d["score"]; tr = np.array(d["transform"], float)
    from rasterio.transform import Affine, rowcol as rc2
    aff = Affine(*tr)
    gj = json.load(open(os.path.join(VIEWER, "known_sites.geojson"), encoding="utf-8"))
    sites = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1])
             for f in gj["features"] if not f["properties"].get("mask")]
    xs, ys = warp_xy("EPSG:4326", str(d["crs"]), [s[0] for s in sites], [s[1] for s in sites])
    rows, cols = rc2(aff, xs, ys)
    rows, cols = np.atleast_1d(rows).astype(int), np.atleast_1d(cols).astype(int)
    site_scores = []
    Hs, Ws = sc.shape
    for r, c in zip(rows, cols):
        if 0 <= r < Hs and 0 <= c < Ws:
            win = sc[max(0, r-2):r+3, max(0, c-2):c+3]
            if np.isfinite(win).any():
                site_scores.append(np.nanmax(win))
    site_scores = np.array(site_scores)
    fin = sc[np.isfinite(sc)]
    rng = np.random.default_rng(7)
    bg = fin[rng.choice(len(fin), size=min(20000, len(fin)), replace=False)]
    obs, ranks, n1, n2 = auc_ranksum(site_scores, bg)

    # null: AUC if n1 labels were assigned at random among all N ranks
    N = n1 + n2; M = 20000
    sums = np.array([rng.choice(ranks, size=n1, replace=False).sum() for _ in range(M)])
    null = (sums - n1 * (n1 + 1) / 2) / (n1 * n2)
    z = (obs - null.mean()) / null.std()
    p = float((null >= obs).mean())
    print(f"(2) PERMUTATION NULL — observed AUC={obs:.3f}  null={null.mean():.3f}+/-{null.std():.3f}  "
          f"z={z:.1f}  p={p:.1e}  (n_site={n1}, n_bg={n2}, M={M})")

    # ---- figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="#11161c")
    for a in ax:
        a.set_facecolor("#0e151c"); a.tick_params(colors="#8da2b5")
        for sp_ in a.spines.values(): sp_.set_color("#283642")
    swir = (wl >= 1950) & (wl <= 2450)
    for s in specs:
        ax[0].plot(wl[swir], s["sp"][swir], color=s["color"], lw=1.8, label=s["name"])
    for x0, lab in ((2200, "Al-OH 2200"), (2345, "carbonate 2345")):
        ax[0].axvline(x0, color="#445566", ls=":", lw=1)
        ax[0].text(x0, ax[0].get_ylim()[1], lab, color="#8da2b5", fontsize=7, rotation=90, va="top", ha="right")
    ax[0].set_xlabel("wavelength (nm)", color="#8da2b5"); ax[0].set_ylabel("reflectance", color="#8da2b5")
    ax[0].set_title("(1) Real EnMAP SWIR spectra — diagnostic absorptions", color="#e8eef4")
    ax[0].legend(fontsize=7.5, facecolor="#141b22", labelcolor="#e8eef4", edgecolor="#283642")

    ax[1].hist(null, bins=60, color="#34c8b4", alpha=.8)
    ax[1].axvline(obs, color="#ff4d4d", lw=2, label=f"observed AUC={obs:.2f}")
    ax[1].set_xlabel("AUC under shuffled labels", color="#8da2b5")
    ax[1].set_title(f"(2) Permutation null — z={z:.0f}, p<{max(p,1/M):.0e}", color="#e8eef4")
    ax[1].legend(fontsize=8, facecolor="#141b22", labelcolor="#e8eef4", edgecolor="#283642")
    fig.tight_layout(); fig.savefig(os.path.join(VIEWER, "verify.png"), dpi=140, facecolor=fig.get_facecolor())

    json.dump({"spectral": [{"name": s["name"], "kind": s["kind"], "color": s["color"],
                             "carbonate_2345_depth": round(s["carb2345"], 4),
                             "alOH_2200_depth": round(s["alOH2200"], 4)} for s in specs],
               "permutation": {"observed_auc": round(float(obs), 4), "null_mean": round(float(null.mean()), 4),
                               "null_std": round(float(null.std()), 4), "z": round(float(z), 1),
                               "p_value": p, "p_floor": 1.0/M, "n_site": int(n1), "n_bg": int(n2), "M": M},
               "run": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(VIEWER, "verify.json"), "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] wrote viewer/verify.png + verify.json")


if __name__ == "__main__":
    main()
