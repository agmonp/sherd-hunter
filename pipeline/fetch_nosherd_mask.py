"""
fetch_nosherd_mask.py — POLYGONS where there cannot be archaeological sherd scatter, so the
detector's hot pixels there can be auto-labelled (not "unvetted"). Israel bbox.

Source: OpenStreetMap / Overpass (open). Ways with inline geometry ('out geom'); each closed
way -> a polygon. Categories (property 'kind'):
  water        natural=water (Dead Sea, reservoirs as water)
  salt_pond    landuse=salt_pond / man_made=salt_pond  (Dead Sea Works evaporation ponds = the
               bright evaporite that dominates the firing screen)
  reservoir    landuse=reservoir / landuse=basin
  urban        landuse=residential / commercial / retail   (modern built-up)
  industrial   landuse=industrial (complements the point-mask in fetch_known_sites)

Out: viewer/nosherd_mask.geojson  (Polygon features; properties.kind / color)
Run: python pipeline/fetch_nosherd_mask.py
"""
import os, json, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
BBOX = (29.45, 34.20, 33.35, 35.92)          # S,W,N,E (Israel)

CATS = {   # kind -> (list of (k,v) tag selectors, colour)
    "water":      ([("natural", "water")], "#2b6cff"),
    "salt_pond":  ([("landuse", "salt_pond"), ("man_made", "salt_pond")], "#d98cff"),
    "reservoir":  ([("landuse", "reservoir"), ("landuse", "basin")], "#3aa0d8"),
    "urban":      ([("landuse", "residential"), ("landuse", "commercial"), ("landuse", "retail")], "#888c95"),
    "industrial": ([("landuse", "industrial")], "#7aa0ff"),
}


def overpass(q, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.post(MIRRORS[i % len(MIRRORS)], data={"data": q}, timeout=240,
                              headers={"User-Agent": "sherdhunter-research/0.1"})
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:100]}"
        except Exception as e:
            last = str(e)
        time.sleep(3)
    raise RuntimeError(f"overpass failed: {last}")


def rel_polys(data, kind, color):
    """Stitch a relation's member ways (out geom) into polygons via shapely.polygonize —
    needed for big multipolygon water bodies like the Dead Sea (a relation, not a way)."""
    from shapely.geometry import LineString, mapping
    from shapely.ops import polygonize, unary_union
    out = []
    for el in data.get("elements", []):
        if el.get("type") != "relation":
            continue
        lines = []
        for m in el.get("members", []):
            g = m.get("geometry")
            if g and len(g) >= 2:
                lines.append(LineString([(p["lon"], p["lat"]) for p in g]))
        if not lines:
            continue
        try:
            polys = list(polygonize(unary_union(lines)))
        except Exception:
            continue
        nm = el.get("tags", {}).get("name", "")
        for poly in polys:
            if poly.area > 0:
                out.append({"type": "Feature", "properties": {"kind": kind, "color": color, "name": nm},
                            "geometry": mapping(poly)})
    return out


def main():
    t0 = time.time()
    s, w, n, e = BBOX
    feats, counts = [], {}
    for kind, (tags, color) in CATS.items():
        sel = "".join(f'way["{k}"="{v}"]({s},{w},{n},{e});' for k, v in tags)
        q = f"[out:json][timeout:180];({sel});out geom;"
        try:
            data = overpass(q)
        except RuntimeError as ex:
            print(f"  {kind:11s} SKIP ({ex})"); counts[kind] = 0; continue
        added = 0
        for el in data.get("elements", []):
            g = el.get("geometry")
            if not g or len(g) < 4:
                continue
            ring = [[round(p["lon"], 6), round(p["lat"], 6)] for p in g]
            if ring[0] != ring[-1]:
                ring.append(ring[0])                     # close the ring
            feats.append({"type": "Feature",
                          "properties": {"kind": kind, "color": color,
                                         "name": el.get("tags", {}).get("name", "")},
                          "geometry": {"type": "Polygon", "coordinates": [ring]}})
            added += 1
        counts[kind] = added
        print(f"  {kind:11s} {added:5d}  ({time.time()-t0:.0f}s)")

    # big water bodies (Dead Sea, Med) + salt-works are RELATIONS -> fetch + polygonize
    for kind, (tags, color) in (("water", CATS["water"]), ("salt_pond", CATS["salt_pond"])):
        sel = "".join(f'relation["{k}"="{v}"]({s},{w},{n},{e});' for k, v in tags)
        q = f"[out:json][timeout:180];({sel});out geom;"
        try:
            rels = rel_polys(overpass(q), kind, color)
        except RuntimeError as ex:
            print(f"  {kind}(rel)  SKIP ({ex})"); continue
        feats.extend(rels); counts[kind + "_rel"] = len(rels)
        print(f"  {kind+'(rel)':11s} {len(rels):5d}  ({time.time()-t0:.0f}s)")

    out = os.path.join(VIEWER, "nosherd_mask.geojson")
    json.dump({"type": "FeatureCollection",
               "meta": {"source": "OpenStreetMap / Overpass", "bbox": list(BBOX),
                        "counts": counts, "run": time.strftime("%Y-%m-%d %H:%M"),
                        "note": "areas with no possible sherd scatter (water/salt-pond/reservoir/"
                                "urban/industrial) -> auto-label detector hotspots, not 'unvetted'"},
               "features": feats},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[{time.time()-t0:.0f}s] wrote {out}: {len(feats)} polygons  {counts}")


if __name__ == "__main__":
    main()
