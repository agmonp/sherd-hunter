"""
detect.py — full-scene / AOI fired-ceramic ACE detection on a real EMIT L2A cube.
Turns the viewer's demo heatmap into a real one.

Usage
  python pipeline/detect.py <granule_id> [--nc path] [--bbox W,S,E,N] [--full]

Defaults: granule = Tel Arad scene _015, bbox = the viewer demo box (35.05,31.23,35.20,31.33).
--full scores the entire swath (row-chunked; raw cube held in RAM, ~1.8 GB float32).

Outputs (viewer/):
  detection.png            RGBA heatmap (transparent off-swath / low score)
  detection_bounds.json    {bbox:[W,S,E,N], granule, stats}  -> viewer flips DEMO->LIVE
  findings.geojson         top off-site hot spots (candidate anomalies)

Memory strategy (16 GB laptop):
  AOI mode reads ONLY the raw slab the GLT references for the bbox (a few hundred MB max).
  Stats (mean/cov) come from a random sample; scoring is chunked.

Honest scope: at 60 m the product is statistical separation + anomaly flagging, NOT a sherd
map. Any hot spot must be checked against natural carbonate/clay outcrops before excitement.
"""
import os, sys, json, time
import numpy as np

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
SCENES = os.path.join(ROOT, "scenes")

TEL_ARAD = (35.1247, 31.2803)                  # lon, lat
KNOWN_SITES = [TEL_ARAD]                       # excluded from "novel anomaly" findings
DEFAULT_BBOX = (35.05, 31.23, 35.20, 31.33)    # W,S,E,N — matches the viewer demo box
WV_WINDOWS = [(1340.0, 1465.0), (1790.0, 1960.0)]  # deep water-vapor bands (nm)
FILL_MAX = -0.005                              # EMIT fill is -9999 / -0.01


# ---------------------------------------------------------------- load (AOI slab)
def open_emit(nc_path):
    import netCDF4 as nc
    ds = nc.Dataset(nc_path)
    wl  = np.array(ds["sensor_band_parameters"]["wavelengths"][:], dtype=np.float64)
    glt_x = np.array(ds["location"]["glt_x"][:], dtype=np.int32)   # (H,W) ortho -> raw col (1-based)
    glt_y = np.array(ds["location"]["glt_y"][:], dtype=np.int32)
    gt = np.array(ds.geotransform, dtype=np.float64)               # ulx,xres,rot,uly,rot,yres(<0)
    return ds, wl, glt_x, glt_y, gt


def ortho_window(gt, glt_shape, bbox):
    """ortho-grid row/col window covering bbox (W,S,E,N)."""
    ulx, xres, _, uly, _, yres = gt
    H, W = glt_shape
    W_, S_, E_, N_ = bbox
    c0 = int(np.clip(np.floor((W_ - ulx) / xres), 0, W - 1))
    c1 = int(np.clip(np.ceil ((E_ - ulx) / xres), 1, W))
    r0 = int(np.clip(np.floor((N_ - uly) / yres), 0, H - 1))      # yres<0: north row first
    r1 = int(np.clip(np.ceil ((S_ - uly) / yres), 1, H))
    if r1 <= r0 or c1 <= c0:
        sys.exit("bbox does not intersect this granule's swath")
    return r0, r1, c0, c1


def load_aoi(ds, glt_x, glt_y, gt, bbox):
    """Read only the raw slab the GLT references inside bbox; assemble ortho cube."""
    r0, r1, c0, c1 = ortho_window(gt, glt_x.shape, bbox)
    gx, gy = glt_x[r0:r1, c0:c1], glt_y[r0:r1, c0:c1]
    valid = (gx > 0) & (gy > 0)
    if not valid.any():
        sys.exit("bbox falls entirely outside the swath (GLT all fill)")
    ry0, ry1 = int(gy[valid].min()) - 1, int(gy[valid].max())     # GLT is 1-indexed
    rx0, rx1 = int(gx[valid].min()) - 1, int(gx[valid].max())
    slab = np.array(ds["reflectance"][ry0:ry1, rx0:rx1, :], dtype=np.float32)
    H, W = gx.shape
    cube = np.full((H, W, slab.shape[-1]), np.nan, np.float32)
    jj, ii = np.nonzero(valid)
    cube[jj, ii, :] = slab[gy[valid] - 1 - ry0, gx[valid] - 1 - rx0, :]
    ulx, xres, _, uly, _, yres = gt
    lon = ulx + (np.arange(c0, c1) + 0.5) * xres
    lat = uly + (np.arange(r0, r1) + 0.5) * yres
    return cube, lon, lat


def run_full(granule_id, nc_path, t0, endmember_name="empirical"):
    """Whole-swath scan, RAM-safe: raw cube once (~1.8 GB f32), ortho assembled and ACE-scored
    in row chunks (never materializes the ~9 GB full ortho cube)."""
    ds, wl, glt_x, glt_y, gt = open_emit(nc_path)
    raw = np.array(ds["reflectance"][:], dtype=np.float32)
    raw[raw <= FILL_MAX] = np.nan
    D, C, B = raw.shape
    print(f"[{time.time()-t0:5.1f}s] raw cube {raw.shape} in RAM")

    # DESTRIPE (critical): EMIT is a pushbroom — each crosstrack column is one detector
    # element with its own slight calibration. An empirical endmember inherits its columns'
    # fingerprint and ACE then "detects" those columns along the whole 140 km track
    # (proven on this scene: stripe = cols 519-524 = the Tel Arad columns).
    # Fix: multiplicative per-column, per-band gain to the scene-wide median.
    col_med = np.nanmedian(raw[::4, :, :], axis=0)            # (C,B) robust column response
    glob_med = np.nanmedian(col_med, axis=0)                  # (B,)
    gain = glob_med[None, :] / np.clip(col_med, 1e-4, None)
    gain = np.clip(gain, 0.8, 1.25).astype(np.float32)        # sane bounds; fill stays NaN
    raw *= gain[None, :, :]
    print(f"[{time.time()-t0:5.1f}s] destriped (per-column gain, median |1-g|="
          f"{np.nanmedian(np.abs(1-gain)):.4f})")
    flat = raw.reshape(-1, B)

    # band mask + background stats from a random sample
    rng = np.random.default_rng(7)
    samp = flat[rng.choice(D * C, size=min(300_000, D * C), replace=False)]
    samp = samp[np.isfinite(samp).any(axis=1)]
    keep = np.isnan(samp).mean(axis=0) <= 0.5
    sk = samp[:, keep]
    sk = sk[np.isfinite(sk).all(axis=1)]
    mean = sk.mean(axis=0, dtype=np.float64)
    cov_inv = np.linalg.inv(np.cov((sk - mean).astype(np.float64), rowvar=False)
                            + 1e-8 * np.eye(int(keep.sum())))
    print(f"[{time.time()-t0:5.1f}s] stats from {len(sk):,} sample px, {int(keep.sum())} bands kept")

    if endmember_name == "empirical":
        # empirical: small box at Tel Arad (kept-band space). CIRCULAR at the tell — use for
        # cross-site transfer only.
        embox = (TEL_ARAD[0] - 0.01, TEL_ARAD[1] - 0.01, TEL_ARAD[0] + 0.01, TEL_ARAD[1] + 0.01)
        cube_em, lon_em, lat_em = load_aoi(ds, glt_x, glt_y, gt, embox)
        cube_em[cube_em <= FILL_MAX] = np.nan
        tgt, em_px = endmember(cube_em[:, :, keep], lon_em, lat_em, *TEL_ARAD)
    else:
        tgt = library_endmember(endmember_name, wl)[keep]
        em_px = []
        print(f"[{time.time()-t0:5.1f}s] LIBRARY endmember '{endmember_name}' (non-circular)")
    t = (tgt - mean).astype(np.float64)
    tCt = float(t @ cov_inv @ t)

    H, W = glt_x.shape
    score = np.full((H, W), np.nan, np.float32)
    n_ok = 0
    for r0 in range(0, H, 250):
        r1 = min(r0 + 250, H)
        gx, gy = glt_x[r0:r1, :], glt_y[r0:r1, :]
        v = (gx > 0) & (gy > 0)
        if not v.any():
            continue
        spec = flat[(gy[v].astype(np.int64) - 1) * C + (gx[v].astype(np.int64) - 1)][:, keep]
        okp = np.isfinite(spec).all(axis=1)
        sc = np.full(int(v.sum()), np.nan, np.float32)
        if okp.any():
            Xc = (spec[okp] - mean).astype(np.float64)
            XC = Xc @ cov_inv
            num = (XC @ t) ** 2
            den = tCt * np.einsum("ij,ij->i", XC, Xc)
            sc[okp] = (num / np.clip(den, 1e-12, None)).astype(np.float32)
            n_ok += int(okp.sum())
        block = np.full(v.shape, np.nan, np.float32)
        block[v] = sc
        score[r0:r1, :] = block
    print(f"[{time.time()-t0:5.1f}s] ACE scored {n_ok:,} valid px (full swath)")

    ulx, xres, _, uly, _, yres = gt
    lon = ulx + (np.arange(W) + 0.5) * xres
    lat = uly + (np.arange(H) + 0.5) * yres
    return score, lon, lat, em_px


def mask_bands(cube, wl):
    """Data-driven: EMIT stores scene-wide fill in ~41 water-vapor bands whose exact
    extent varies (e.g. 1327-1432, 1774-1960 nm) — wider than the nominal windows.
    Drop any band that is fill in >50% of in-swath pixels."""
    cube[cube <= FILL_MAX] = np.nan
    flat = cube.reshape(-1, cube.shape[-1])
    in_swath = np.isfinite(flat).any(axis=1)
    nanfrac = np.isnan(flat[in_swath]).mean(axis=0)
    bad = nanfrac > 0.5
    return cube[:, :, ~bad], wl[~bad]


# ---------------------------------------------------------------- detector
def library_endmember(name, wl_full):
    """Load a lab spectrum from pipeline/spectral_library.json (USGS splib07a, resampled
    to the EMIT grid) and interpolate onto wl_full. NON-CIRCULAR: detector never sees the site."""
    import json as _json
    lib = _json.load(open(os.path.join(ROOT, "pipeline", "spectral_library.json")))
    if name not in lib["spectra"]:
        sys.exit(f"endmember '{name}' not in library; have: {', '.join(lib['spectra'])}")
    wl_lib = np.array(lib["wl_nm"], dtype=np.float64)
    return np.interp(wl_full, wl_lib, np.array(lib["spectra"][name], dtype=np.float64))


def endmember(cube, lon, lat, plon, plat, k=5):
    """Empirical fired-ceramic/anthrosol signature: mean of k nearest valid pixels to a point."""
    LON, LAT = np.meshgrid(lon, lat)
    ok = np.isfinite(cube).all(axis=2)
    d = (LON - plon) ** 2 + (LAT - plat) ** 2
    d[~ok] = np.inf
    idx = np.unravel_index(np.argsort(d, axis=None)[:k], d.shape)
    return np.nanmean(cube[idx[0], idx[1], :], axis=0), list(zip(idx[0].tolist(), idx[1].tolist()))


def ace_full(cube, target, sample=120_000, chunk=200_000, rng_seed=7):
    """Chunked ACE over all valid pixels. Returns score grid in [0,1] (NaN off-swath)."""
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    ok = np.isfinite(X).all(axis=1)
    n_ok = int(ok.sum())
    rng = np.random.default_rng(rng_seed)
    samp = rng.choice(np.nonzero(ok)[0], size=min(sample, n_ok), replace=False)
    mean = X[samp].mean(axis=0, dtype=np.float64)
    cov = np.cov((X[samp] - mean).astype(np.float64), rowvar=False) + 1e-8 * np.eye(B)
    cov_inv = np.linalg.inv(cov)
    t = (target - mean).astype(np.float64)
    tCt = float(t @ cov_inv @ t)

    score = np.full(H * W, np.nan, np.float32)
    idx_ok = np.nonzero(ok)[0]
    for s in range(0, len(idx_ok), chunk):
        sel = idx_ok[s:s + chunk]
        Xc = (X[sel] - mean).astype(np.float64)
        XC = Xc @ cov_inv
        num = (XC @ t) ** 2
        den = tCt * np.einsum("ij,ij->i", XC, Xc)
        score[sel] = (num / np.clip(den, 1e-12, None)).astype(np.float32)
    return score.reshape(H, W), n_ok


# ---------------------------------------------------------------- outputs
STOPS = [(0.0,(40,30,90)),(0.25,(30,120,200)),(0.5,(40,200,140)),
         (0.7,(230,220,40)),(0.85,(240,140,30)),(1.0,(220,30,30))]

def render_png(score_disp, path):
    """Vectorized RGBA render. score_disp in [0,1], NaN -> transparent. Row 0 = north."""
    from PIL import Image
    v = score_disp.copy()
    nan = ~np.isfinite(v)
    v[nan] = 0.0
    xs = [s[0] for s in STOPS]
    img = np.zeros(v.shape + (4,), np.uint8)
    for ch in range(3):
        img[..., ch] = np.interp(v, xs, [s[1][ch] for s in STOPS]).astype(np.uint8)
    alpha = (np.clip((v - 0.15) / 0.5, 0, 1) * 200).astype(np.uint8)
    alpha[nan] = 0
    img[..., 3] = alpha
    Image.fromarray(img).save(path)


def export(score, lon, lat, granule_id, em_px, endmember_name="empirical"):
    p999 = float(np.nanpercentile(score, 99.9))
    disp = np.clip(score / max(p999, 1e-6), 0, 1)
    render_png(disp, os.path.join(VIEWER, "detection.png"))
    np.savez_compressed(os.path.join(ROOT, "scenes", f"score_{endmember_name}.npz"),
                        score=score.astype(np.float32),
                        lon0=float(lon.min()), lon1=float(lon.max()),
                        lat0=float(lat.min()), lat1=float(lat.max()))

    W_, E_ = float(lon.min()), float(lon.max())
    S_, N_ = float(lat.min()), float(lat.max())
    stats = {"valid_px": int(np.isfinite(score).sum()),
             "p50": float(np.nanpercentile(score, 50)),
             "p99": float(np.nanpercentile(score, 99)),
             "p999": p999, "max": float(np.nanmax(score))}
    json.dump({"granule": granule_id, "bbox": [W_, S_, E_, N_],
               "label": f"REAL ACE detection (endmember: {endmember_name})", "stats": stats,
               "endmember": endmember_name, "endmember_px": em_px,
               "run": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(VIEWER, "detection_bounds.json"), "w"), indent=1)

    # findings: strongest pixels, deduped on a ~0.003 deg grid, off known sites, capped
    thr = float(np.nanpercentile(score, 99.9))
    jj, ii = np.nonzero(np.nan_to_num(score) >= thr)
    order = np.argsort(score[jj, ii])[::-1]
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
                      "properties": {"score": round(float(score[r, c]), 4),
                                     "disp": round(float(disp[r, c]), 3),
                                     "on_known_site": on_site},
                      "geometry": {"type": "Point", "coordinates": [round(plon, 5), round(plat, 5)]}})
        if len(feats) >= 50:
            break
    json.dump({"type": "FeatureCollection", "granule": granule_id, "features": feats},
              open(os.path.join(VIEWER, "findings.geojson"), "w"), indent=1)
    return stats, feats


def run(granule_id, nc_path=None, bbox=DEFAULT_BBOX, full=False, endmember_name="empirical"):
    t0 = time.time()
    nc_path = nc_path or os.path.join(SCENES, granule_id + ".nc")
    if not os.path.exists(nc_path):
        sys.exit(f"cube not found: {nc_path} — download it first (see README)")

    if full:
        score, lon, lat, em_px = run_full(granule_id, nc_path, t0, endmember_name)
    else:
        ds, wl, glt_x, glt_y, gt = open_emit(nc_path)
        print(f"[{time.time()-t0:5.1f}s] opened; ortho grid {glt_x.shape}, {len(wl)} bands")
        cube, lon, lat = load_aoi(ds, glt_x, glt_y, gt, bbox)
        print(f"[{time.time()-t0:5.1f}s] cube {cube.shape}  lon {lon.min():.3f}..{lon.max():.3f}  lat {lat.min():.3f}..{lat.max():.3f}")
        cube, wl = mask_bands(cube, wl)
        tgt, em_px = endmember(cube, lon, lat, *TEL_ARAD)
        print(f"[{time.time()-t0:5.1f}s] bands kept {cube.shape[-1]}; endmember from {len(em_px)} px @ Tel Arad")
        score, n_ok = ace_full(cube, tgt)
        print(f"[{time.time()-t0:5.1f}s] ACE scored {n_ok:,} valid px")

    stats, feats = export(score, lon, lat, granule_id, em_px, endmember_name)
    off = sum(1 for f in feats if not f["properties"]["on_known_site"])
    print(f"[{time.time()-t0:5.1f}s] DONE  max={stats['max']:.3f} p99.9={stats['p999']:.4f}  "
          f"findings: {len(feats)} ({off} off-site)")
    print("viewer will now show the REAL heatmap (badge flips to LIVE)")


if __name__ == "__main__":
    args = sys.argv[1:]
    gid = args[0] if args and not args[0].startswith("--") else "EMIT_L2A_RFL_001_20250616T104810_2516707_015"
    nc  = args[args.index("--nc") + 1] if "--nc" in args else None
    bb  = tuple(float(x) for x in args[args.index("--bbox") + 1].split(",")) if "--bbox" in args else DEFAULT_BBOX
    em  = args[args.index("--endmember") + 1] if "--endmember" in args else "empirical"
    run(gid, nc, bb, full="--full" in args, endmember_name=em)
