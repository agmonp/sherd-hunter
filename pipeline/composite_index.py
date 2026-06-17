"""
composite_index.py — full-swath PHYSICS-FEATURE COMPOSITE index (replaces ACE as primary screen).

Why: ROC analysis (roc_analysis.py) showed spectral matching fails or cheats here
(ACE_brick AUC 0.42, ACE_empirical 0.92 but circular), while three direction-locked
physical features genuinely separate Tel Arad from matched terrain:
    carbonate 2345 nm CR depth   AUC 0.963   (+ deeper on tell)
    Al-OH    2200 nm CR depth    AUC 0.877   (- SHALLOWER on tell: firing signature)
    brightness 500-1300 nm       AUC 0.841   (+ brighter on tell)

Composite (NO training, directions locked a priori from session-1 physics):
    z_i = robust z-score of feature i over all valid swath pixels (median/MAD)
    composite = ( z_carb  -  z_alOH  +  z_bright ) / 3
Vegetation (NDVI >= 0.25) masked out. Destriped identically to detect.py.

Outputs
  viewer/detection.png + detection_bounds.json (method=COMPOSITE -> viewer badge)
  viewer/findings.geojson      top off-site candidates (deduped, UNVETTED)
  scenes/score_composite.npz   raw composite grid (ortho), ROC-ready
Also prints full-swath AUC validation vs the matched annulus used in roc_analysis.py.

Run: python pipeline/composite_index.py [--nc path]
"""
import os, sys, json, time, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from detect import open_emit, render_png, TEL_ARAD, KNOWN_SITES, FILL_MAX

VIEWER = os.path.join(ROOT, "viewer")
SCENES = os.path.join(ROOT, "scenes")
NC_DEFAULT = os.path.join(SCENES, "EMIT_L2A_RFL_001_20250616T104810_2516707_015.nc")

R_TELL, R_HALO, R_BG = 0.250, 1.500, 5.000     # km — same design as roc_analysis.py
KM_PER_DEG = 111.0
WEIGHTS = {"carbonate": +1.0, "alOH": -1.0, "brightness": +1.0}   # LOCKED, no training

warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Mean of empty slice")


def cr_depth_raw(raw, wl, w0, w1):
    """Continuum-removed depth on the RAW (downtrack,crosstrack,band) grid.
    Linear continuum anchored at window edges; depth = 1 - min(R/continuum)."""
    m = (wl >= w0) & (wl <= w1)
    w = wl[m]
    S = raw[:, :, m]
    a, b = S[:, :, 0], S[:, :, -1]
    t = (w - w[0]) / (w[-1] - w[0])
    cont = a[..., None] + (b - a)[..., None] * t[None, None, :]
    return 1.0 - np.nanmin(S / np.clip(cont, 1e-6, None), axis=2)


def robust_z(x, valid, clip=4.0):
    """Median/MAD z, winsorized at +-clip so single pathological pixels (shadow/water
    blow-ups in the CR ratio) cannot dominate the composite."""
    med = float(np.median(x[valid]))
    mad = float(np.median(np.abs(x[valid] - med)))
    return np.clip((x - med) / max(1.4826 * mad, 1e-9), -clip, clip), med, mad


def local_background(g, block=16, smooth_iters=2):
    """Coarse local median background: nanmedian over (block x block) px tiles
    (16 px = ~1 km at 60 m), then nan-aware 3x3 tile smoothing (x2 -> ~5 km support),
    nearest-upsampled back. Subtracting it turns the composite into a LOCAL anomaly —
    the full-swath analog of the validated tell-vs-its-own-annulus ROC design, and it
    cancels broad terrain-class offsets (the geology problem). Tile wrap at np.roll
    edges only touches off-swath NaN margins."""
    H, W = g.shape
    Hp, Wp = -(-H // block) * block, -(-W // block) * block
    p = np.full((Hp, Wp), np.nan, np.float32)
    p[:H, :W] = g
    b = np.nanmedian(p.reshape(Hp // block, block, Wp // block, block), axis=(1, 3))
    for _ in range(smooth_iters):
        acc = np.zeros_like(b)
        cnt = np.zeros_like(b)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sh = np.roll(np.roll(b, dy, 0), dx, 1)
                m = np.isfinite(sh)
                acc[m] += sh[m]
                cnt[m] += 1
        nxt = np.full_like(b, np.nan)
        nz = cnt > 0
        nxt[nz] = acc[nz] / cnt[nz]
        b = nxt
    return np.repeat(np.repeat(b, block, 0), block, 1)[:H, :W]


def auc(pos, neg):
    v = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-v); y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.trapezoid(tpr, fpr))


def main(nc_path):
    t0 = time.time()
    ds, wl, glt_x, glt_y, gt = open_emit(nc_path)
    raw = np.array(ds["reflectance"][:], dtype=np.float32)
    raw[raw <= FILL_MAX] = np.nan
    print(f"[{time.time()-t0:5.1f}s] raw cube {raw.shape}")

    # destripe — identical to detect.py run_full (per-column multiplicative gain)
    col_med = np.nanmedian(raw[::4, :, :], axis=0)
    glob_med = np.nanmedian(col_med, axis=0)
    gain = np.clip(glob_med[None, :] / np.clip(col_med, 1e-4, None), 0.8, 1.25).astype(np.float32)
    raw *= gain[None, :, :]
    print(f"[{time.time()-t0:5.1f}s] destriped")

    # ---- features on the raw grid (D,C)
    def nb(target):
        return int(np.argmin(np.abs(wl - target)))
    carb   = cr_depth_raw(raw, wl, 2300, 2375)
    aloh   = cr_depth_raw(raw, wl, 2120, 2245)
    bright = np.nanmean(raw[:, :, (wl >= 500) & (wl <= 1300)], axis=2)
    red, nir = raw[:, :, nb(660)], raw[:, :, nb(850)]
    ndvi = (nir - red) / np.clip(nir + red, 1e-6, None)
    del raw
    print(f"[{time.time()-t0:5.1f}s] features computed (carb/alOH/brightness/NDVI)")

    valid = np.isfinite(carb) & np.isfinite(aloh) & np.isfinite(bright) & np.isfinite(ndvi)
    # dark pixels (shadow/water, refl<0.05): CR continuum anchors near zero make the
    # ratio explode (observed z up to +6e4) — physically meaningless, mask them out
    keep = valid & (ndvi < 0.25) & (bright >= 0.05)
    z_carb, *_ = robust_z(carb, keep)
    z_aloh, *_ = robust_z(aloh, keep)
    z_brig, *_ = robust_z(bright, keep)
    comp = (WEIGHTS["carbonate"] * z_carb
            + WEIGHTS["alOH"] * z_aloh
            + WEIGHTS["brightness"] * z_brig) / 3.0
    comp[~keep] = np.nan
    print(f"[{time.time()-t0:5.1f}s] composite on {int(keep.sum()):,} raw px "
          f"(veg-masked {int((valid & ~keep).sum()):,})")

    # ---- orthorectify composite + per-feature grids via GLT
    H, W = glt_x.shape
    v = (glt_x > 0) & (glt_y > 0)
    def to_ortho(g):
        o = np.full((H, W), np.nan, np.float32)
        o[v] = g[glt_y[v] - 1, glt_x[v] - 1]
        return o
    comp_o, carb_o, aloh_o, brig_o = map(to_ortho, (comp, carb, aloh, bright))
    ulx, xres, _, uly, _, yres = gt
    lon = ulx + (np.arange(W) + 0.5) * xres
    lat = uly + (np.arange(H) + 0.5) * yres
    print(f"[{time.time()-t0:5.1f}s] ortho grids {comp_o.shape}, "
          f"{int(np.isfinite(comp_o).sum()):,} valid px")

    # LOCAL ANOMALY = composite minus ~5 km median background (primary screen)
    anom_o = comp_o - local_background(comp_o)
    print(f"[{time.time()-t0:5.1f}s] local-anomaly composite ready")

    # ---- validation against the matched-annulus design (same as roc_analysis.py)
    LON, LAT = np.meshgrid(lon, lat)
    dist_km = np.sqrt(((LON - TEL_ARAD[0]) * KM_PER_DEG * np.cos(np.radians(TEL_ARAD[1]))) ** 2
                      + ((LAT - TEL_ARAD[1]) * KM_PER_DEG) ** 2)
    fin = np.isfinite(anom_o)
    tell = fin & (dist_km <= R_TELL)
    bg = fin & (dist_km >= R_HALO) & (dist_km <= R_BG)
    aucs = {}
    for name, grid, sign in (("composite", comp_o, +1), ("anomaly", anom_o, +1),
                             ("carbonate", carb_o, +1), ("alOH", aloh_o, -1),
                             ("brightness", brig_o, +1)):
        aucs[name] = round(auc(grid[tell] * sign, grid[bg] * sign), 4)
    print(f"[{time.time()-t0:5.1f}s] AUC vs matched annulus: " +
          "  ".join(f"{k}={v:.3f}" for k, v in aucs.items()) +
          f"   (tell n={int(tell.sum())}, bg n={int(bg.sum())})")

    # tell rank among ALL valid swath pixels — for BOTH variants (honesty check)
    ranks = {}
    for name, grid in (("composite", comp_o), ("anomaly", anom_o)):
        gv = grid[np.isfinite(grid)]
        tb = float(np.nanmax(grid[tell])) if tell.any() else float("nan")
        ranks[name] = {"tell_best": round(tb, 3),
                       "rank": int((gv > tb).sum()) + 1, "n": int(len(gv)),
                       "pct": round(100.0 * float((gv <= tb).mean()), 3)}
        print(f"   {name:9s} best tell px {tb:+.3f} -> rank {ranks[name]['rank']:,} "
              f"of {len(gv):,} ({ranks[name]['pct']:.3f} pctile)")
    vals = anom_o[fin]

    # ---- exports (same files the viewer auto-loads; anomaly composite REPLACES ACE)
    p999 = float(np.nanpercentile(vals, 99.9))
    disp = np.clip(anom_o / max(p999, 1e-6), 0, 1)
    render_png(disp, os.path.join(VIEWER, "detection.png"))
    np.savez_compressed(os.path.join(SCENES, "score_composite.npz"),
                        score=anom_o.astype(np.float32),
                        composite_raw=comp_o.astype(np.float32),
                        lon0=float(lon.min()), lon1=float(lon.max()),
                        lat0=float(lat.min()), lat1=float(lat.max()))

    stats = {"valid_px": int(fin.sum()), "p50": float(np.nanpercentile(vals, 50)),
             "p99": float(np.nanpercentile(vals, 99)), "p999": p999,
             "max": float(np.nanmax(vals)), "auc": aucs, "tell_rank": ranks}
    json.dump({"granule": os.path.basename(nc_path).replace(".nc", ""),
               "bbox": [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())],
               "method": "COMPOSITE",
               "label": "Local-anomaly composite physics index: +carbonate2345 -AlOH2200 "
                        "+brightness (robust-z winsorized, locked directions, no training; "
                        "~5 km median background removed; veg+dark masked, destriped)",
               "stats": stats, "run": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(VIEWER, "detection_bounds.json"), "w"), indent=1)

    # findings: top ANOMALY pixels deduped on ~330 m grid, known sites flagged, capped at 50
    thr = p999
    jj, ii = np.nonzero(np.nan_to_num(anom_o, nan=-9e9) >= thr)
    order = np.argsort(anom_o[jj, ii])[::-1]
    feats, seen = [], set()
    for k in order:
        r, c = int(jj[k]), int(ii[k])
        plon, plat = float(lon[c]), float(lat[r])
        cell = (round(plon / 0.003), round(plat / 0.003))
        if cell in seen:
            continue
        seen.add(cell)
        on_site = any((plon - kx) ** 2 + (plat - ky) ** 2 < 0.01 ** 2 for kx, ky in KNOWN_SITES)
        feats.append({"type": "Feature",
                      "properties": {"score": round(float(anom_o[r, c]), 4),
                                     "disp": round(float(disp[r, c]), 3),
                                     "on_known_site": on_site},
                      "geometry": {"type": "Point", "coordinates": [round(plon, 5), round(plat, 5)]}})
        if len(feats) >= 50:
            break
    json.dump({"type": "FeatureCollection",
               "granule": os.path.basename(nc_path).replace(".nc", ""),
               "method": "COMPOSITE", "features": feats},
              open(os.path.join(VIEWER, "findings.geojson"), "w"), indent=1)
    off = sum(1 for f in feats if not f["properties"]["on_known_site"])
    print(f"[{time.time()-t0:5.1f}s] DONE  composite max={stats['max']:+.3f} p99.9={p999:+.3f}  "
          f"findings: {len(feats)} ({off} off-site, UNVETTED)")


if __name__ == "__main__":
    args = sys.argv[1:]
    nc = args[args.index("--nc") + 1] if "--nc" in args else NC_DEFAULT
    main(nc)
