"""
validate_against_known.py — cross-reference the composite anomaly grid with KNOWN layers.

Answers three questions for the current scene's composite index:
  1. Do KNOWN archaeological sites score high?  -> AUC(known-site px vs random background),
     plus each in-swath site's swath-percentile. The honest generalization test: these sites
     were NOT used to build the index (only Tel Arad's disk was, in the ROC study).
  2. Which composite HOTSPOTS are explained by modern industry?  -> for each top candidate,
     distance to nearest OSM industrial/quarry/works (mask) feature.
  3. Which hotspots are NOVEL?  -> high score, far from any known site AND any mask feature.

Inputs : scenes/score_composite.npz  (from composite_index.py)
         viewer/known_sites.geojson   (from fetch_known_sites.py)
Outputs: viewer/validation.json + console table
Run    : python pipeline/validate_against_known.py
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
SCENES = os.path.join(ROOT, "scenes")
KM_PER_DEG = 111.0


def load_grid():
    p = os.path.join(SCENES, "score_composite.npz")
    if not os.path.exists(p):
        sys.exit("score_composite.npz not found — run composite_index.py first")
    d = np.load(p)
    sc = d["score"]
    lon0, lon1 = float(d["lon0"]), float(d["lon1"])
    lat0, lat1 = float(d["lat0"]), float(d["lat1"])
    H, W = sc.shape
    return sc, lon0, lon1, lat0, lat1, H, W


def rc_of(plon, plat, lon0, lon1, lat0, lat1, H, W):
    c = int(round((plon - lon0) / (lon1 - lon0) * (W - 1)))
    r = int(round((lat1 - plat) / (lat1 - lat0) * (H - 1)))   # row 0 = north
    return r, c


def sample_max(sc, r, c, rad=3):
    H, W = sc.shape
    if not (0 <= r < H and 0 <= c < W):
        return np.nan
    win = sc[max(0, r-rad):r+rad+1, max(0, c-rad):c+rad+1]
    return float(np.nanmax(win)) if np.isfinite(win).any() else np.nan


def auc(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-v); y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.trapezoid(tpr, fpr))


def main():
    t0 = time.time()
    sc, lon0, lon1, lat0, lat1, H, W = load_grid()
    fin = np.isfinite(sc)
    finvals = sc[fin]
    pad = 0.01  # require sites a hair inside the bounds
    gj = json.load(open(os.path.join(VIEWER, "known_sites.geojson"), encoding="utf-8"))

    sites, masks = [], []
    for f in gj.get("features", []):
        lon, lat = f["geometry"]["coordinates"]
        if not (lon0+pad <= lon <= lon1-pad and lat0+pad <= lat <= lat1-pad):
            continue
        (masks if f["properties"].get("mask") else sites).append((lon, lat, f["properties"]))
    print(f"in-swath: {len(sites)} known archaeological features, {len(masks)} mask features")

    def pctile(v):
        return 100.0 * float((finvals <= v).mean()) if np.isfinite(v) else np.nan

    # 1) known archaeological sites: percentile + AUC vs random background
    site_scores = np.array([sample_max(sc, *rc_of(lon, lat, lon0, lon1, lat0, lat1, H, W))
                            for lon, lat, _ in sites])
    rng = np.random.default_rng(7)
    bg = finvals[rng.choice(len(finvals), size=min(20000, len(finvals)), replace=False)]
    site_auc = auc(site_scores, bg)
    med_pct = float(np.nanmedian([pctile(s) for s in site_scores])) if len(sites) else float("nan")
    print(f"known-site composite: AUC(site vs random)={site_auc:.3f}  "
          f"median site percentile={med_pct:.1f}")

    # named in-swath sites, ranked
    named = sorted([(s, sample_max(sc, *rc_of(lon, lat, lon0, lon1, lat0, lat1, H, W)), lon, lat, p)
                    for (lon, lat, p), s in zip(sites, site_scores)],
                   key=lambda x: (-(x[1] if np.isfinite(x[1]) else -9))) if False else None

    site_rows = []
    for (lon, lat, p), s in zip(sites, site_scores):
        site_rows.append({"name": p.get("name") or p.get("name_he") or p.get("site_type") or "?",
                          "category": p.get("category"), "lon": round(lon, 4), "lat": round(lat, 4),
                          "score": None if not np.isfinite(s) else round(s, 3),
                          "pct": None if not np.isfinite(s) else round(pctile(s), 1)})
    site_rows.sort(key=lambda r: (r["pct"] is None, -(r["pct"] or 0)))
    print("\n  top known sites by composite percentile:")
    for r in site_rows[:12]:
        nm = (r["name"][:34]) if r["name"] else "?"
        print(f"    {nm:34s} {str(r['pct']):>6}pct  score={r['score']}  ({r['category']})")

    # 2+3) top composite hotspots: classify near-known / near-industrial / novel
    def nearest_km(plon, plat, pts):
        if not pts:
            return None, 9e9
        best, bd = None, 9e9
        for lon, lat, pp in pts:
            d = np.hypot((plon-lon)*KM_PER_DEG*np.cos(np.radians(plat)), (plat-lat)*KM_PER_DEG)
            if d < bd:
                bd, best = d, pp
        return best, bd

    thr = float(np.nanpercentile(finvals, 99.9))
    jj, ii = np.nonzero(np.nan_to_num(sc, nan=-9e9) >= thr)
    order = np.argsort(sc[jj, ii])[::-1]
    seen, cands = set(), []
    for k in order:
        r, c = int(jj[k]), int(ii[k])
        plon = lon0 + c/(W-1)*(lon1-lon0)
        plat = lat1 - r/(H-1)*(lat1-lat0)
        cell = (round(plon/0.01), round(plat/0.01))
        if cell in seen:
            continue
        seen.add(cell)
        sp, sd = nearest_km(plon, plat, sites)
        mp, md = nearest_km(plon, plat, masks)
        klass = ("industrial?" if md <= 1.0 else
                 "near known site" if sd <= 1.0 else
                 "NOVEL candidate")
        cands.append({"lon": round(plon, 4), "lat": round(plat, 4),
                      "score": round(float(sc[r, c]), 3),
                      "near_site_km": round(sd, 1) if sd < 9e8 else None,
                      "near_site": (sp.get("name") or sp.get("name_he")) if sp else None,
                      "near_mask_km": round(md, 1) if md < 9e8 else None,
                      "near_mask": (mp.get("name") or mp.get("category")) if mp else None,
                      "class": klass})
        if len(cands) >= 30:
            break
    nnov = sum(1 for c in cands if c["class"] == "NOVEL candidate")
    nind = sum(1 for c in cands if c["class"] == "industrial?")
    nkno = sum(1 for c in cands if c["class"] == "near known site")
    print(f"\n  top-{len(cands)} hotspots: {nkno} near known sites, {nind} likely industrial, "
          f"{nnov} NOVEL")
    for c in cands[:12]:
        tag = {"NOVEL candidate": "NOVEL", "industrial?": "indus", "near known site": "known"}[c["class"]]
        ref = c["near_site"] if c["class"] == "near known site" else c["near_mask"] if c["class"]=="industrial?" else f"site {c['near_site_km']}km"
        print(f"    {c['lat']:.4f}N {c['lon']:.4f}E  s={c['score']:+.2f}  [{tag}] {ref or ''}")

    out = {"scene_grid": "score_composite.npz",
           "n_sites_in_swath": len(sites), "n_mask_in_swath": len(masks),
           "site_auc_vs_random": None if not np.isfinite(site_auc) else round(site_auc, 4),
           "median_site_percentile": None if not np.isfinite(med_pct) else round(med_pct, 1),
           "sites": site_rows, "candidates": cands, "run": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open(os.path.join(VIEWER, "validation.json"), "w"), indent=1)
    print(f"\n[{time.time()-t0:.0f}s] wrote viewer/validation.json")


if __name__ == "__main__":
    main()
