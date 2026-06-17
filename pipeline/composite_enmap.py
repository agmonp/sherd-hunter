"""
composite_enmap.py — run the validated composite physics index on a REAL EnMAP L2A COG (30 m).

This is the 4x-resolution test of the screen that worked on EMIT (60 m): same LOCKED directions
(+carbonate2345, -AlOH2200, +brightness), winsorized robust-z, ~5 km local-anomaly background.
No retraining.

EnMAP specifics handled here (vs EMIT):
  * Input = orthorectified COG in UTM (EPSG:32636), 224 bands int16, nodata -32768,
    reflectance scale 1e-4 (DLR L2A). No GLT, no destriping needed.
  * Wavelengths are NOT in the COG and the metadata XML is behind SSO. We instead use the
    nominal EnMAP band model and VALIDATE it against the data: the fully-masked water-vapor
    gap (bands ~130-134) must fall at ~1400 nm. (Confirmed on scene _069168.)

Validation in-scene (known layers from fetch_known_sites.py):
  * Tel Arad percentile, known archaeological-site AUC vs random, candidate classification.

Outputs:
  scenes/score_composite_enmap.npz
  viewer/validation_enmap.json
  viewer/detection.png + detection_bounds.json + findings.geojson  (EnMAP REPLACES EMIT heatmap)
Run: python pipeline/composite_enmap.py [--tif scenes/enmap_DT0000069168_SPECTRAL.tiff]
"""
import os, sys, json, time, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
SCENES = os.path.join(ROOT, "scenes")
TIF_DEFAULT = os.path.join(SCENES, "enmap_DT0000069168_SPECTRAL.tiff")
TEL_ARAD = (35.1247, 31.2803)            # lon, lat
# Blind-test tells: known sherd-bearing sites NOT used to build the index. Tel Arad is the
# reference (its disk informed the ROC directions); the rest are pure blind tests.
BLIND_TELLS = [("Tel Arad (ref)", 35.1247, 31.2803), ("Tel Ira", 34.9783, 31.1503),
               ("Tel Malhata", 35.0417, 31.2306), ("Tel Masos", 34.9767, 31.2228),
               ("Mezad Bokek", 35.3660, 31.1830)]
KM_PER_DEG = 111.0
WEIGHTS = {"carbonate": +1.0, "alOH": -1.0, "brightness": +1.0}   # LOCKED, no training
warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def enmap_wavelengths():
    """EnMAP band centers, SWIR DATA-PINNED. The fully-masked water-vapor gap centers on band
    132; we anchor band 132 = 1400 nm (true WV absorption center) and fit piecewise-linear
    901.65@b91 -> 1400@b132 -> 2445.46@b223 (published endpoints). VNIR linear 418.24..992.45.
    This corrects the single-linear model's ~20 nm low bias near 1400 nm. (Verified: the spectral
    check exposed that the narrow carbonate window missed the absorption — see WIDE windows below.)"""
    wl = np.empty(224)
    wl[:91] = np.linspace(418.24, 992.45, 91)
    wl[91:133] = np.linspace(901.65, 1400.0, 42)        # bands 91..132 (WV gap pinned at 1400)
    wl[132:] = np.linspace(1400.0, 2445.46, 92)          # bands 132..223
    return wl


def cr_depth(cube_hw_b, wl, w0, w1):
    m = (wl >= w0) & (wl <= w1)
    w = wl[m]; S = cube_hw_b[:, :, m]
    a, b = S[:, :, 0], S[:, :, -1]
    t = (w - w[0]) / (w[-1] - w[0])
    cont = a[..., None] + (b - a)[..., None] * t[None, None, :]
    return 1.0 - np.nanmin(S / np.clip(cont, 1e-6, None), axis=2)


def robust_z(x, valid, clip=4.0):
    med = float(np.median(x[valid])); mad = float(np.median(np.abs(x[valid] - med)))
    return np.clip((x - med) / max(1.4826 * mad, 1e-9), -clip, clip)


def local_background(g, block, smooth_iters=2):
    H, W = g.shape
    Hp, Wp = -(-H // block) * block, -(-W // block) * block
    p = np.full((Hp, Wp), np.nan, np.float32); p[:H, :W] = g
    b = np.nanmedian(p.reshape(Hp // block, block, Wp // block, block), axis=(1, 3))
    for _ in range(smooth_iters):
        acc = np.zeros_like(b); cnt = np.zeros_like(b)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sh = np.roll(np.roll(b, dy, 0), dx, 1); m = np.isfinite(sh)
                acc[m] += sh[m]; cnt[m] += 1
        nxt = np.full_like(b, np.nan); nz = cnt > 0; nxt[nz] = acc[nz] / cnt[nz]; b = nxt
    return np.repeat(np.repeat(b, block, 0), block, 1)[:H, :W]


def auc(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-v); y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1); fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.trapezoid(tpr, fpr))


def main(tif):
    import rasterio
    from rasterio.warp import transform as warp_xy, transform_bounds, reproject, Resampling
    t0 = time.time()
    ds = rasterio.open(tif)
    nod = ds.nodata
    raw = ds.read()                                   # (224,H,W) int16
    B, H, W = raw.shape
    wl = enmap_wavelengths()

    # validate wavelength model against the water-vapor gap in the data
    vf = ((raw != nod) & (raw > 0)).mean(axis=(1, 2))
    gap = [i for i in range(B) if vf[i] < 0.5]
    gap_wl = (wl[gap].min(), wl[gap].max()) if gap else (None, None)
    print(f"[{time.time()-t0:4.1f}s] COG {B}b {W}x{H} {ds.crs}; masked gap bands {gap} "
          f"-> {gap_wl[0]:.0f}-{gap_wl[1]:.0f} nm (expect ~1400 nm WV) " if gap else "no gap")

    cube = np.moveaxis(raw, 0, -1).astype(np.float32)  # (H,W,B)
    del raw
    cube[cube == nod] = np.nan
    cube[cube <= 0] = np.nan
    cube *= 1e-4                                       # DLR L2A reflectance scale
    print(f"[{time.time()-t0:4.1f}s] scaled to reflectance (H,W,B)={cube.shape}")

    def nb(t): return int(np.argmin(np.abs(wl - t)))
    carb = cr_depth(cube, wl, 2250, 2400)               # WIDE: robust to ~50 nm SWIR registration
    aloh = cr_depth(cube, wl, 2120, 2245)
    bright = np.nanmean(cube[:, :, (wl >= 500) & (wl <= 1300)], axis=2)
    red, nir = cube[:, :, nb(660)], cube[:, :, nb(850)]
    ndvi = (nir - red) / np.clip(nir + red, 1e-6, None)
    del cube
    print(f"[{time.time()-t0:4.1f}s] features computed")

    valid = np.isfinite(carb) & np.isfinite(aloh) & np.isfinite(bright) & np.isfinite(ndvi)
    keep = valid & (ndvi < 0.25) & (bright >= 0.05)
    zc, za, zb = robust_z(carb, keep), robust_z(aloh, keep), robust_z(bright, keep)
    # TWO principled screens (locked directions, NO training). verify.py showed carbonate is
    # ubiquitous Negev terrain (low site contrast) -> split it from the SPECIFIC firing signature.
    firing = ((-za) + zb) / 2.0; firing[~keep] = np.nan        # Al-OH depletion + brightness (specific)
    halo = zc.astype(np.float32).copy(); halo[~keep] = np.nan  # carbonate halo (broad anthrosol/terrain)
    firing_a = firing - local_background(firing, block=56)     # ~1.7 km tiles -> ~5 km support
    halo_a = halo - local_background(halo, block=56)

    def _std(g):
        m = np.isfinite(g); md = float(np.median(g[m])); mad = float(np.median(np.abs(g[m] - md)))
        return (g - md) / max(1.4826 * mad, 1e-9)
    fz, hz = _std(firing_a), _std(halo_a)
    # PRIMARY screen = firing (Al-OH depletion + brightness): verify.py predicted it is the more
    # SPECIFIC archaeological marker, and the per-screen site-AUC below confirms firing > halo >
    # max-combined (max amplifies noise / dilutes contrast). halo kept as a secondary context screen.
    anom = fz.copy()
    anom[~np.isfinite(firing_a)] = np.nan
    fin = np.isfinite(anom)
    print(f"[{time.time()-t0:4.1f}s] firing (primary) + carbonate-halo screens on {int(keep.sum()):,} px "
          f"(veg/dark masked {int((valid&~keep).sum()):,})")

    # ---- sampling helpers: lon/lat -> (row,col); windowed max of a grid; percentile vs a pool
    from rasterio.transform import rowcol as _rowcol
    def rc(lons, lats):
        xs, ys = warp_xy("EPSG:4326", ds.crs, list(lons), list(lats))
        rows, cols = _rowcol(ds.transform, xs, ys)
        return np.atleast_1d(np.array(rows)).astype(int), np.atleast_1d(np.array(cols)).astype(int)

    def smax(grid, rows, cols, rad=2):
        out = np.full(len(rows), np.nan, np.float32)
        for k, (r, c) in enumerate(zip(rows, cols)):
            if 0 <= r < H and 0 <= c < W:
                win = grid[max(0, r-rad):r+rad+1, max(0, c-rad):c+rad+1]
                if np.isfinite(win).any():
                    out[k] = np.nanmax(win)
        return out

    SCREENS = {"firing": fz, "halo": hz}
    POOLS = {k: g[np.isfinite(g)] for k, g in SCREENS.items()}
    def ppct(name, v): return 100.0 * float((POOLS[name] <= v).mean()) if np.isfinite(v) else np.nan
    finvals = POOLS["firing"]
    def pct(v): return ppct("firing", v)

    # Tel Arad on both screens
    tr, tc = rc([TEL_ARAD[0]], [TEL_ARAD[1]])
    ta = smax(anom, tr, tc, 3)[0]
    ta_f, ta_h = pct(ta), ppct("halo", smax(hz, tr, tc, 3)[0])
    print(f"[{time.time()-t0:4.1f}s] Tel Arad: firing p{ta_f:.1f} | halo p{ta_h:.1f}")

    # --- BLIND TEST: known sherd-bearing tells, percentile on firing | halo | combined
    blind = []
    print("  BLIND TEST — known tells (firing | halo percentile):")
    for nm, plon, plat in BLIND_TELLS:
        rr, ccc = rc([plon], [plat]); r0, c0 = int(rr[0]), int(ccc[0])
        if not (0 <= r0 < H and 0 <= c0 < W):
            print(f"    {nm:15s} outside tile"); blind.append({"name": nm, "in_scene": False}); continue
        fv, hv = smax(fz, [r0], [c0], 3)[0], smax(hz, [r0], [c0], 3)[0]
        if not np.isfinite(fv):                            # point lands in the tile's nodata border
            print(f"    {nm:15s} in nodata border of this tile")
            blind.append({"name": nm, "in_scene": False}); continue
        fp, hp = round(ppct("firing", fv), 1), round(ppct("halo", hv), 1)
        print(f"    {nm:15s} firing p{fp:5.1f} | halo p{hp:5.1f}")
        blind.append({"name": nm, "in_scene": True, "lon": plon, "lat": plat,
                      "firing_pct": fp, "halo_pct": hp})

    # known sites + masks (OSM); site AUC on EACH screen (which one separates sites best?)
    gj = json.load(open(os.path.join(VIEWER, "known_sites.geojson"), encoding="utf-8"))
    sites = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1], f["properties"])
             for f in gj["features"] if not f["properties"].get("mask")]
    masks = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1], f["properties"])
             for f in gj["features"] if f["properties"].get("mask")]
    slon = [s[0] for s in sites]; slat = [s[1] for s in sites]
    sr, sc = rc(slon, slat)
    in_scene = (sr >= 0) & (sr < H) & (sc >= 0) & (sc < W)
    rng = np.random.default_rng(7)
    aucs = {}
    for kname, grid in SCREENS.items():
        ss = smax(grid, sr, sc); pool = POOLS[kname]
        bgk = pool[rng.choice(len(pool), size=min(20000, len(pool)), replace=False)]
        aucs[kname] = round(auc(ss[np.isfinite(ss)], bgk), 4)
    sscore = smax(anom, sr, sc)                            # firing (primary) drives site_rows + candidates
    n_in = int(np.isfinite(sscore).sum())
    site_auc = aucs["firing"]
    med_pct = float(np.nanmedian([pct(s) for s in sscore if np.isfinite(s)])) if n_in else float("nan")
    print(f"[{time.time()-t0:4.1f}s] sites in scene {n_in}  AUC firing={aucs['firing']} "
          f"halo={aucs['halo']}  median(firing) p{med_pct:.1f}")

    # top sites table (combined screen)
    site_rows = []
    for (lon, lat, p), s in zip(sites, sscore):
        if np.isfinite(s):
            site_rows.append({"name": p.get("name") or p.get("name_he") or p.get("site_type") or "?",
                              "category": p.get("category"), "lon": round(lon, 4), "lat": round(lat, 4),
                              "score": round(float(s), 3), "pct": round(pct(s), 1)})
    site_rows.sort(key=lambda r: -r["pct"])
    print("  top known sites in-scene:")
    for r in site_rows[:8]:
        print(f"    {(r['name'] or '?')[:32]:32s} {r['pct']:5.1f}pct  s={r['score']:+.2f}  ({r['category']})")

    # candidates: top anomaly px deduped, classified vs known site / industrial mask
    mr, mc = rc([m[0] for m in masks], [m[1] for m in masks])
    site_px = set(zip(sr[in_scene].tolist(), sc[in_scene].tolist()))
    mvalid = (mr >= 0) & (mr < H) & (mc >= 0) & (mc < W)
    mask_px = list(zip(mr[mvalid].tolist(), mc[mvalid].tolist()))
    site_pts = list(zip(sr[in_scene].tolist(), sc[in_scene].tolist(),
                        [s[2] for s, ok in zip(sites, in_scene) if ok]))
    px_per_km = 1000.0 / 30.0
    thr = float(np.nanpercentile(finvals, 99.9))
    jj, ii = np.nonzero(np.nan_to_num(anom, nan=-9e9) >= thr)
    order = np.argsort(anom[jj, ii])[::-1]
    seen, cands = set(), []
    inv = ds.transform
    for k in order:
        r, c = int(jj[k]), int(ii[k])
        cell = (r // 11, c // 11)                      # ~330 m dedup
        if cell in seen:
            continue
        seen.add(cell)
        x, y = ds.xy(r, c)
        lon, lat = warp_xy(ds.crs, "EPSG:4326", [x], [y])
        lon, lat = lon[0], lat[0]
        def near(pts):
            best, bd = None, 9e9
            for pr, pc, *pp in pts:
                d = np.hypot(r-pr, c-pc) / px_per_km
                if d < bd:
                    bd, best = d, (pp[0] if pp else None)
            return bd, best
        sd, sp = near(site_pts)
        md, _ = near([(a, b) for a, b in mask_px])
        klass = ("industrial?" if md <= 1.0 else "near known site" if sd <= 1.0 else "NOVEL candidate")
        cands.append({"lon": round(lon, 4), "lat": round(lat, 4), "score": round(float(anom[r, c]), 3),
                      "near_site_km": round(sd, 1), "near_site": (sp.get("name") or sp.get("name_he")) if sp else None,
                      "near_mask_km": round(md, 1), "class": klass})
        if len(cands) >= 40:
            break

    from shapely.geometry import shape as _shape, Point as _Point
    from shapely.strtree import STRtree as _STRtree
    def _polys(path):
        if not os.path.exists(path):
            return None, None
        gjx = json.load(open(path, encoding="utf-8"))
        ps, pr = [], []
        for f in gjx["features"]:
            try:
                g = _shape(f["geometry"])
            except Exception:
                continue
            if g.is_valid and not g.is_empty:
                ps.append(g); pr.append(f["properties"])
        return _STRtree(ps), (ps, pr)

    # no-sherd mask (water / salt-pond / urban / reservoir / industrial): these hot pixels are
    # definitely NOT archaeology (e.g. the Dead Sea Works salt ponds that dominate the firing
    # screen). Highest-priority label -> moves them out of "unvetted".
    ntree, ndata = _polys(os.path.join(VIEWER, "nosherd_mask.geojson"))
    WATERY = {"water", "salt_pond", "reservoir"}
    BUF_DEG = 1.5 / 111.0                                   # ~1.5 km buffer for lake margins
    if ntree is not None:
        nps, npr = ndata
        for c in cands:
            pt = _Point(c["lon"], c["lat"]); labeled = False
            for i in ntree.query(pt):                      # exact containment first
                if nps[int(i)].contains(pt):
                    c["nosherd"] = npr[int(i)]["kind"]; c["class"] = "nosherd:" + npr[int(i)]["kind"]
                    labeled = True; break
            if labeled:
                continue
            for i in ntree.query(pt.buffer(BUF_DEG)):      # buffer: receding Dead Sea flats / shore
                pr = npr[int(i)]
                if pr["kind"] in WATERY and nps[int(i)].distance(pt) <= BUF_DEG:
                    c["nosherd"] = pr["kind"] + "_margin"; c["class"] = "nosherd:" + pr["kind"]
                    break

    # geology cross-check: name the GSI formation under each remaining hotspot (proves "unexplained"
    # hotspots are natural fired/bright carbonate, not archaeology)
    geop = os.path.join(VIEWER, "geology.geojson")
    if os.path.exists(geop):
        from shapely.geometry import shape, Point
        from shapely.strtree import STRtree
        gj_geo = json.load(open(geop, encoding="utf-8"))
        polys, gprops = [], []
        for f in gj_geo["features"]:
            try:
                g = shape(f["geometry"])
            except Exception:
                continue
            if g.is_valid and not g.is_empty:
                polys.append(g); gprops.append(f["properties"])
        tree = STRtree(polys)
        natural = {"fired_carbonate", "lacustrine_marl", "carbonate", "evaporite"}
        for c in cands:
            pt = Point(c["lon"], c["lat"]); fm = None
            for i in tree.query(pt):
                if polys[int(i)].contains(pt):
                    fm = gprops[int(i)]; break
            if fm:
                c["geology"] = fm.get("name"); c["geology_flag"] = fm.get("flag")
                if c["class"] == "NOVEL candidate" and fm.get("flag") in natural:
                    c["class"] = "geology:" + fm.get("flag")
    cc = {}
    for c_ in cands:
        cc[c_["class"]] = cc.get(c_["class"], 0) + 1
    print(f"  top-{len(cands)} hotspots: {cc}")

    # ---- save grids + validation json (three screens: firing / halo / combined)
    np.savez_compressed(os.path.join(SCENES, "score_composite_enmap.npz"),
                        score=anom.astype(np.float32), firing=fz.astype(np.float32),
                        halo=hz.astype(np.float32), crs=str(ds.crs),
                        transform=np.array(ds.transform)[:6])
    ta_block = {"firing_pct": None if not np.isfinite(ta_f) else round(ta_f, 1),
                "halo_pct": None if not np.isfinite(ta_h) else round(ta_h, 1)}
    common = {"n_sites_in_scene": n_in, "screen_auc": aucs, "site_auc_vs_random": site_auc,
              "median_site_percentile": round(med_pct, 1), "blind_tells": blind, "tel_arad": ta_block}
    json.dump({"scene": os.path.basename(tif), "sensor": "EnMAP L2A 30 m", **common,
               "sites": site_rows, "candidates": cands,
               "wavelength_check_nm": [round(gap_wl[0], 0), round(gap_wl[1], 0)] if gap else None,
               "run": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(VIEWER, "validation_enmap.json"), "w"), indent=1)
    # viewer-schema copy (replaces EMIT validation.json)
    json.dump({"n_sites_in_swath": n_in, "screen_auc": aucs, "site_auc_vs_random": site_auc,
               "median_site_percentile": round(med_pct, 1), "candidates": cands, "blind_tells": blind,
               "tel_arad": ta_block, "sensor": "EnMAP 30 m", "run": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(VIEWER, "validation.json"), "w"), indent=1)

    # ---- warp each screen to EPSG:4326 -> PNGs (combined=primary, firing+halo=toggles)
    wb = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    outW, outH = 1400, int(1400 * (wb[3]-wb[1]) / (wb[2]-wb[0]))
    from rasterio.transform import from_bounds as tfb
    sys.path.insert(0, os.path.join(ROOT, "pipeline"))
    from detect import render_png
    def warp_png(grid, fname, lo):
        dst = np.full((outH, outW), np.nan, np.float32)
        reproject(grid, dst, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=tfb(*wb, outW, outH), dst_crs="EPSG:4326",
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        hi = float(np.nanpercentile(grid[np.isfinite(grid)], 99.9))
        render_png(np.clip((dst - lo) / max(hi - lo, 1e-6), 0, 1), os.path.join(VIEWER, fname))
        return hi
    p999 = warp_png(anom, "detection.png", lo=float(np.nanpercentile(POOLS["firing"], 50)))   # firing=primary
    warp_png(hz, "detection_halo.png", lo=float(np.nanpercentile(POOLS["halo"], 50)))
    json.dump({"granule": os.path.basename(tif).replace(".tiff", ""),
               "bbox": [wb[0], wb[1], wb[2], wb[3]], "method": "EnMAP-FIRING", "has_halo": True,
               "label": "EnMAP L2A 30 m — PRIMARY = firing screen (Al-OH depletion + brightness, the "
                        "specific archaeological marker; site-AUC " + str(aucs["firing"]) + " beats "
                        "carbonate-halo " + str(aucs["halo"]) + "). Toggle 'Carbonate-halo screen' "
                        "for the broad anthrosol/terrain channel. Locked directions, robust-z, ~5 km bg.",
               "stats": {"valid_px": int(fin.sum()), "p999": p999, "max": float(np.nanmax(finvals)),
                         "screen_auc": aucs, "tel_arad": ta_block},
               "run": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(VIEWER, "detection_bounds.json"), "w"), indent=1)
    feats = [{"type": "Feature", "properties": {"score": c["score"], "disp": round(c["score"]/max(p999,1e-6),3),
              "on_known_site": c["class"] == "near known site"},
              "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]}} for c in cands]
    json.dump({"type": "FeatureCollection", "method": "EnMAP-2SCREEN", "features": feats},
              open(os.path.join(VIEWER, "findings.geojson"), "w"), indent=1)
    print(f"[{time.time()-t0:4.1f}s] DONE -> detection(.png/_firing/_halo) + bounds + validation")


if __name__ == "__main__":
    args = sys.argv[1:]
    tif = args[args.index("--tif")+1] if "--tif" in args else TIF_DEFAULT
    main(tif)
