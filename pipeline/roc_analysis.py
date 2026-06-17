"""
roc_analysis.py — the formal test: do tell pixels separate from MATCHED-terrain background?

Design
  positive: pixels within R_TELL of Tel Arad center (EB II tell + lower city, ~40 px @ 60 m)
  excluded: 'halo' R_TELL..R_HALO (ambiguous scatter edge)
  negative: annulus R_HALO..R_BG on the SAME carbonate plateau, minus vegetation (NDVI),
            minus any pixel with missing bands
  features (direction locked a priori from session-1 hypothesis):
    carbonate_2345_depth  (+ deeper on tell: anthropogenic ash/lime)
    alOH_2200_depth       (- shallower on tell: fired-ceramic dehydroxylation)
    brightness            (+ brighter on tell)
    ACE_brick             (+; non-circular USGS lab endmember)
    ACE_empirical         (+; CIRCULAR at the tell — reported but flagged)

All spectra destriped (per-column gain) before features. Single scene; clouds not masked
(L2A MASK file integration pending); tell polygon approximated as a disk — stated caveats.

Run: python pipeline/roc_analysis.py
Out: viewer/roc.json, viewer/roc_analysis.png, console table
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from detect import open_emit, ortho_window, TEL_ARAD, FILL_MAX, DEFAULT_BBOX

VIEWER = os.path.join(ROOT, "viewer")
SCENES = os.path.join(ROOT, "scenes")
NC = os.path.join(SCENES, "EMIT_L2A_RFL_001_20250616T104810_2516707_015.nc")

R_TELL, R_HALO, R_BG = 0.250, 1.500, 5.000   # km
KM_PER_DEG = 111.0


def load_destriped_aoi(bbox):
    t0 = time.time()
    ds, wl, glt_x, glt_y, gt = open_emit(NC)
    raw = np.array(ds["reflectance"][:], dtype=np.float32)
    raw[raw <= FILL_MAX] = np.nan
    col_med = np.nanmedian(raw[::4, :, :], axis=0)
    glob_med = np.nanmedian(col_med, axis=0)
    gain = np.clip(glob_med[None, :] / np.clip(col_med, 1e-4, None), 0.8, 1.25).astype(np.float32)
    raw *= gain[None, :, :]
    print(f"[{time.time()-t0:5.1f}s] raw destriped")

    r0, r1, c0, c1 = ortho_window(gt, glt_x.shape, bbox)
    gx, gy = glt_x[r0:r1, c0:c1], glt_y[r0:r1, c0:c1]
    valid = (gx > 0) & (gy > 0)
    H, W = gx.shape
    cube = np.full((H, W, raw.shape[-1]), np.nan, np.float32)
    jj, ii = np.nonzero(valid)
    cube[jj, ii, :] = raw[gy[valid] - 1, gx[valid] - 1, :]
    del raw
    ulx, xres, _, uly, _, yres = gt
    lon = ulx + (np.arange(c0, c1) + 0.5) * xres
    lat = uly + (np.arange(r0, r1) + 0.5) * yres
    print(f"[{time.time()-t0:5.1f}s] AOI cube {cube.shape}")
    return cube, wl, lon, lat, (r0, r1, c0, c1)


def band_mask(cube, wl):
    flat = cube.reshape(-1, cube.shape[-1])
    in_sw = np.isfinite(flat).any(axis=1)
    nanfrac = np.isnan(flat[in_sw]).mean(axis=0)
    keep = nanfrac <= 0.5
    return cube[:, :, keep], wl[keep]


def nearest_band(wl, target):
    return int(np.argmin(np.abs(wl - target)))


def cr_depth(cube, wl, w0, w1, center):
    """Per-pixel continuum-removed absorption depth: local linear continuum anchored at the
    window edges (standard for narrow diagnostic windows), depth at the band nearest `center`
    region = 1 - min(refl/continuum) inside the window."""
    m = (wl >= w0) & (wl <= w1)
    w = wl[m]; S = cube[:, :, m]                     # (H,W,B)
    a, b = S[:, :, 0], S[:, :, -1]
    t = (w - w[0]) / (w[-1] - w[0])                  # (B,)
    cont = a[..., None] + (b - a)[..., None] * t[None, None, :]
    ratio = S / np.clip(cont, 1e-6, None)
    return 1.0 - np.nanmin(ratio, axis=2)


def main():
    cube, wl, lon, lat, win = load_destriped_aoi(DEFAULT_BBOX)
    cube, wl = band_mask(cube, wl)
    H, W = cube.shape[:2]
    LON, LAT = np.meshgrid(lon, lat)
    dist_km = np.sqrt(((LON - TEL_ARAD[0]) * KM_PER_DEG * np.cos(np.radians(TEL_ARAD[1]))) ** 2
                      + ((LAT - TEL_ARAD[1]) * KM_PER_DEG) ** 2)

    ok = np.isfinite(cube).all(axis=2)
    red = cube[:, :, nearest_band(wl, 660)]
    nir = cube[:, :, nearest_band(wl, 850)]
    ndvi = (nir - red) / np.clip(nir + red, 1e-6, None)

    feats = {
        "carbonate_2345_depth": (cr_depth(cube, wl, 2300, 2375, 2345), +1, False),
        "alOH_2200_depth":      (cr_depth(cube, wl, 2120, 2245, 2200), -1, False),
        "brightness":           (np.nanmean(cube[:, :, (wl >= 500) & (wl <= 1300)], axis=2), +1, False),
    }
    # ACE score grids (full-swath) — slice our window
    r0, r1, c0, c1 = win
    for nm, f in (("ACE_brick", "score_BrickMean_fired_clay.npz"),
                  ("ACE_empirical", "score_empirical.npz")):
        p = os.path.join(SCENES, f)
        if os.path.exists(p):
            g = np.load(p)["score"][r0:r1, c0:c1]
            feats[nm] = (g, +1, nm == "ACE_empirical")

    tell = ok & (dist_km <= R_TELL)
    bg   = ok & (dist_km >= R_HALO) & (dist_km <= R_BG) & (ndvi < 0.25)
    print(f"tell px: {tell.sum()}   matched-background px: {bg.sum()}   (veg excluded: "
          f"{int((ok & (dist_km>=R_HALO) & (dist_km<=R_BG) & (ndvi>=0.25)).sum())})")

    def auc_roc(pos, neg):
        v = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        order = np.argsort(-v); y = y[order]
        tpr = np.cumsum(y) / max(y.sum(), 1)
        fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
        auc = float(np.trapezoid(tpr, fpr))
        ds = np.linspace(0, len(v) - 1, min(len(v), 120)).astype(int)
        return auc, fpr[ds].tolist(), tpr[ds].tolist()

    results, curves = {}, {}
    for name, (grid, sign, circular) in feats.items():
        pos, neg = grid[tell] * sign, grid[bg] * sign
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        auc, fpr, tpr = auc_roc(pos, neg)
        results[name] = {"auc": round(auc, 4), "n_tell": int(len(pos)), "n_bg": int(len(neg)),
                         "median_tell": round(float(np.median(pos) * sign), 5),
                         "median_bg": round(float(np.median(neg) * sign), 5),
                         "direction": "+" if sign > 0 else "-", "circular": circular}
        curves[name] = {"fpr": fpr, "tpr": tpr}
        print(f"  {name:22s} AUC={auc:.3f}  median tell={np.median(pos)*sign:+.4f} "
              f"bg={np.median(neg)*sign:+.4f}  {'[CIRCULAR]' if circular else ''}")

    # sensitivity to tell-disk radius
    sens = {}
    for r in (0.15, 0.35):
        t2 = ok & (dist_km <= r)
        g, sign, _ = feats["carbonate_2345_depth"]
        pos = g[t2] * sign; neg = g[bg] * sign
        a, _, _ = auc_roc(pos[np.isfinite(pos)], neg[np.isfinite(neg)])
        sens[f"carbonate_r{int(r*1000)}m"] = round(a, 4)
    print("sensitivity (carbonate AUC by tell radius):", sens)

    meta = {"scene": os.path.basename(NC), "design":
            f"tell disk {R_TELL}km vs matched annulus {R_HALO}-{R_BG}km, NDVI<0.25, destriped",
            "caveats": ["tell polygon approximated as disk", "single scene",
                        "clouds not masked (L2A MASK pending)",
                        "ACE_empirical endmember taken from tell center -> inflated"],
            "sensitivity": sens, "run": time.strftime("%Y-%m-%d %H:%M")}
    json.dump({"results": results, "curves": curves, "meta": meta},
              open(os.path.join(VIEWER, "roc.json"), "w"))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#11161c")
    for ax in axes:
        ax.set_facecolor("#0e151c"); ax.tick_params(colors="#8da2b5")
        for s in ax.spines.values(): s.set_color("#283642")
    colors = {"carbonate_2345_depth": "#ff7a3c", "alOH_2200_depth": "#34c8b4",
              "brightness": "#f5c542", "ACE_brick": "#b48cf0", "ACE_empirical": "#8da2b5"}
    for name, c in curves.items():
        ls = "--" if results[name]["circular"] else "-"
        axes[0].plot(c["fpr"], c["tpr"], ls, color=colors.get(name, "#fff"), lw=2,
                     label=f"{name}  AUC={results[name]['auc']:.2f}"
                           + (" (circular)" if results[name]["circular"] else ""))
    axes[0].plot([0, 1], [0, 1], ":", color="#445566", lw=1)
    axes[0].set_xlabel("false positive rate", color="#8da2b5")
    axes[0].set_ylabel("true positive rate", color="#8da2b5")
    axes[0].set_title("Tel Arad vs matched terrain — ROC (EMIT 60 m, destriped)", color="#e8eef4")
    axes[0].legend(fontsize=7.5, facecolor="#141b22", labelcolor="#e8eef4", edgecolor="#283642")

    g, sign, _ = feats["carbonate_2345_depth"]
    axes[1].hist(g[bg][np.isfinite(g[bg])], bins=60, density=True, alpha=.75,
                 color="#34c8b4", label=f"matched bg (n={int(bg.sum())})")
    axes[1].hist(g[tell][np.isfinite(g[tell])], bins=24, density=True, alpha=.75,
                 color="#ff7a3c", label=f"tell (n={int(tell.sum())})")
    axes[1].set_xlabel("carbonate 2345 nm CR depth", color="#8da2b5")
    axes[1].set_title("Best physics feature — distributions", color="#e8eef4")
    axes[1].legend(fontsize=8, facecolor="#141b22", labelcolor="#e8eef4", edgecolor="#283642")
    fig.tight_layout()
    fig.savefig(os.path.join(VIEWER, "roc_analysis.png"), dpi=140,
                facecolor=fig.get_facecolor())
    print("wrote viewer/roc.json + viewer/roc_analysis.png")


if __name__ == "__main__":
    main()
