"""
mesma_enmap.py — MESMA-style sub-pixel unmixing on EnMAP, vs the firing screen (pre-registered).

Hypothesis (pre-registered 2026-06-17): a physically-grounded sub-pixel ABUNDANCE
(fraction of a FIRED-CERAMIC endmember) is a better archaeological detector than the heuristic
firing screen. SUCCESS = known-site AUC >= 0.81 (firing screen = 0.776), bootstrap-significant.
Secondary = a calibrated % abundance + a model-fit RMSE quality flag even if AUC only ties.
NULL is a valid outcome: if MESMA does not beat the simple feature, we keep the simple one.

Method: per pixel, fit several endmember MODELS by (approx-NNLS) least squares, pick the lowest
RMSE; fired-abundance = fired fraction of the chosen model (0 if fired not chosen). Endmembers:
  bg_soil   = median of a random matched-terrain sample (scene-derived; NON-circular w.r.t. sites)
  fired     = USGS splib07 BrickMean_fired_clay   (library — detector never sees the tell)
  carbonate = USGS Calcite_WS272
  clay      = USGS Montmorillonite_SWy1
Models: {bg}, {bg,fired}, {bg,carb}, {bg,clay}, {bg,fired,carb}. Same local-anomaly (~5 km) and
same site/background sampling + no-sherd mask as composite_enmap, so the AUC is directly comparable.

Out: viewer/mesma.json (+ detection_mesma.png if it wins). Run: python pipeline/mesma_enmap.py
"""
import os, sys, json, time, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER, SCENES = os.path.join(ROOT, "viewer"), os.path.join(ROOT, "scenes")
TIF = os.path.join(SCENES, "enmap_DT0000069168_SPECTRAL.tiff")
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from composite_enmap import enmap_wavelengths, local_background, robust_z, auc, TEL_ARAD
warnings.filterwarnings("ignore", category=RuntimeWarning)
FIRING_AUC = 0.7759            # the bar to beat (pre-registered)
KM_PER_DEG = 111.0


def fcls_mesma(X, models):
    """X:(N,B) reflectance. models: list of (E[B,k], fired_col or None). Returns fired-fraction
    and best RMSE per pixel (clip-non-negative least squares; FCLS-style approximation)."""
    N = X.shape[0]
    best = np.full(N, np.inf, np.float32)
    fired = np.zeros(N, np.float32)
    for E, fcol in models:
        pinv = np.linalg.pinv(E).astype(np.float32)        # (k,B)
        A = np.clip(X @ pinv.T, 0, None)                   # (N,k) non-negative abundances
        rmse = np.sqrt(np.mean((A @ E.T - X) ** 2, axis=1)).astype(np.float32)
        upd = rmse < best
        best[upd] = rmse[upd]
        fr = (A[:, fcol] / np.clip(A.sum(1), 1e-9, None)) if fcol is not None else np.zeros(N, np.float32)
        fired[upd] = fr[upd]
    return fired, best


def main():
    import rasterio
    from rasterio.warp import transform as warp_xy
    from rasterio.transform import rowcol
    t0 = time.time()
    ds = rasterio.open(TIF); nod = ds.nodata; wl = enmap_wavelengths()
    raw = ds.read(); B, H, W = raw.shape
    cube = np.moveaxis(raw, 0, -1).astype(np.float32); del raw
    cube[cube == nod] = np.nan; cube[cube <= 0] = np.nan; cube *= 1e-4

    # keep clean bands (drop the masked WV gap), build feature masks
    vf = np.mean(np.isfinite(cube), axis=(0, 1))
    kb = vf > 0.6
    def nb(t): return int(np.argmin(np.abs(wl - t)))
    bright = np.nanmean(cube[:, :, (wl >= 500) & (wl <= 1300)], axis=2)
    red, nir = cube[:, :, nb(660)], cube[:, :, nb(850)]
    ndvi = (nir - red) / np.clip(nir + red, 1e-6, None)
    finite = np.isfinite(cube[:, :, kb]).all(axis=2) & np.isfinite(bright) & np.isfinite(ndvi)
    keep = finite & (ndvi < 0.25) & (bright >= 0.05)
    print(f"[{time.time()-t0:4.1f}s] {int(keep.sum()):,} keep px, {int(kb.sum())} clean bands")

    wlk = wl[kb]
    # endmembers (resampled to kept EnMAP bands)
    lib = json.load(open(os.path.join(ROOT, "pipeline", "spectral_library.json")))
    lw = np.array(lib["wl_nm"], float)
    def em(name): return np.interp(wlk, lw, np.array(lib["spectra"][name], float)).astype(np.float32)
    fired_em, carb_em, clay_em = em("BrickMean_fired_clay"), em("Calcite_WS272"), em("Montmorillonite_SWy1")
    rng = np.random.default_rng(7)
    ksamp = cube[:, :, kb][keep][rng.choice(int(keep.sum()), size=min(40000, int(keep.sum())), replace=False)]
    bg_soil = np.nanmedian(ksamp, axis=0).astype(np.float32)
    print(f"[{time.time()-t0:4.1f}s] endmembers ready (bg_soil from {len(ksamp):,} matched px)")

    # models (column index of 'fired' within each, or None)
    def M(*ems): return np.stack(ems, axis=1)             # (B,k)
    models = [(M(bg_soil), None),
              (M(bg_soil, fired_em), 1),
              (M(bg_soil, carb_em), None),
              (M(bg_soil, clay_em), None),
              (M(bg_soil, fired_em, carb_em), 1)]

    # full-scene fired-abundance (chunked to bound memory)
    Xall = cube[:, :, kb].reshape(-1, int(kb.sum()))
    ok = keep.reshape(-1)
    fired_full = np.full(ok.shape, np.nan, np.float32)
    rmse_full = np.full(ok.shape, np.nan, np.float32)
    idx = np.nonzero(ok)[0]
    for s in range(0, len(idx), 200000):
        sel = idx[s:s+200000]
        fr, rm = fcls_mesma(Xall[sel], models)
        fired_full[sel] = fr; rmse_full[sel] = rm
    fired_ab = fired_full.reshape(H, W)
    fired_ab[~keep] = np.nan
    print(f"[{time.time()-t0:4.1f}s] MESMA done; median fired-abundance "
          f"{np.nanmedian(fired_ab[keep]):.3f}, median RMSE {np.nanmedian(rmse_full[ok]):.4f}")

    # local anomaly (same as firing screen) so AUC is comparable
    anom = fired_ab - local_background(fired_ab, block=56)

    # site / background sampling — same protocol as composite_enmap
    def rc(lons, lats):
        xs, ys = warp_xy("EPSG:4326", ds.crs, list(lons), list(lats))
        r, c = rowcol(ds.transform, xs, ys)
        return np.atleast_1d(np.array(r)).astype(int), np.atleast_1d(np.array(c)).astype(int)
    def smax(grid, rr, cc, rad=2):
        out = np.full(len(rr), np.nan, np.float32)
        for k, (r, c) in enumerate(zip(rr, cc)):
            if 0 <= r < H and 0 <= c < W:
                win = grid[max(0, r-rad):r+rad+1, max(0, c-rad):c+rad+1]
                if np.isfinite(win).any():
                    out[k] = np.nanmax(win)
        return out

    gj = json.load(open(os.path.join(VIEWER, "known_sites.geojson"), encoding="utf-8"))
    sites = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1])
             for f in gj["features"] if not f["properties"].get("mask")]
    sr, sc = rc([s[0] for s in sites], [s[1] for s in sites])
    fin = np.isfinite(anom); pool = anom[fin]
    bg = pool[rng.choice(len(pool), size=min(20000, len(pool)), replace=False)]

    def auc_ci(grid):
        ss = smax(grid, sr, sc); ss = ss[np.isfinite(ss)]
        bb = grid[np.isfinite(grid)]; bsamp = bb[rng.choice(len(bb), size=min(20000, len(bb)), replace=False)]
        a = auc(ss, bsamp)
        boot = []
        for _ in range(500):
            s2 = ss[rng.integers(0, len(ss), len(ss))]
            b2 = bsamp[rng.integers(0, len(bsamp), 2000)]
            boot.append(auc(s2, b2))
        return a, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(ss)

    mesma_auc, lo, hi, nsite = auc_ci(anom)
    # Tel Arad / Tel Malhata abundance
    def pct(v): return 100.0 * float((pool <= v).mean()) if np.isfinite(v) else float("nan")
    diag = {}
    for nmn, lon, lat in [("Tel Arad", TEL_ARAD[0], TEL_ARAD[1]), ("Tel Malhata", 35.0417, 31.2306)]:
        rr, cc = rc([lon], [lat])
        ab = smax(fired_ab, rr, cc, 3)[0]; an = smax(anom, rr, cc, 3)[0]
        diag[nmn] = {"fired_abundance": None if not np.isfinite(ab) else round(float(ab), 3),
                     "anom_pct": None if not np.isfinite(an) else round(pct(an), 1)}

    verdict = ("WIN" if mesma_auc >= 0.81 else "tie" if abs(mesma_auc - FIRING_AUC) <= 0.02 else "loss")
    print(f"[{time.time()-t0:4.1f}s] MESMA fired-abundance site-AUC = {mesma_auc:.3f} "
          f"[95% {lo:.3f}-{hi:.3f}]  vs firing {FIRING_AUC}  -> {verdict}")
    for k, v in diag.items():
        print(f"    {k}: fired-abundance {v['fired_abundance']}  anom p{v['anom_pct']}")

    json.dump({"sensor": "EnMAP 30 m", "n_sites": nsite,
               "mesma_site_auc": round(mesma_auc, 4), "mesma_auc_ci95": [round(lo, 4), round(hi, 4)],
               "firing_auc": FIRING_AUC, "preregistered_win_threshold": 0.81, "verdict": verdict,
               "median_fired_abundance": round(float(np.nanmedian(fired_ab[keep])), 4),
               "median_rmse": round(float(np.nanmedian(rmse_full[ok])), 5),
               "diagnostics": diag, "endmembers": ["bg_soil(scene)", "BrickMean_fired_clay",
               "Calcite_WS272", "Montmorillonite_SWy1"], "run": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(VIEWER, "mesma.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(SCENES, "score_mesma_enmap.npz"),
                        fired_abundance=fired_ab.astype(np.float32), anom=anom.astype(np.float32),
                        crs=str(ds.crs), transform=np.array(ds.transform)[:6])
    print(f"[{time.time()-t0:4.1f}s] wrote viewer/mesma.json + scenes/score_mesma_enmap.npz")


if __name__ == "__main__":
    main()
