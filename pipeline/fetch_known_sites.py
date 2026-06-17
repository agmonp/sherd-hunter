"""
fetch_known_sites.py — pull KNOWN-KNOWLEDGE layers for Israel into the viewer.

Two roles in SherdHunter:
  1. VALIDATION  — overlay known archaeological sites on the composite-index heatmap.
                   Known sites scoring high = true positives; high scores far from any
                   known site = novel candidates worth follow-up.
  2. FALSE-POSITIVE MASK — modern industrial / quarry / built footprints explain the
                   non-archaeological hotspots the index finds (e.g. the Rotem phosphate
                   plant, Dead Sea Works). Flag them so we don't chase them.

Source: OpenStreetMap via the Overpass API (open, no auth, immediate). This is NOT the
Israel Antiquities Authority official database (discover.iaa.org.il, ~4M records, no open
geo-API) — but for MAJOR known sites (tells, ruins) OSM overlaps it well, and it is
reproducible. The IAA official layer is a future drop-in if/when we obtain API access; the
viewer reads whatever this writes, so swapping the source later needs no UI change.

Start scope: ISRAEL (bbox below). Expand by editing BBOX / adding categories.

Out: viewer/known_sites.geojson  (FeatureCollection; each feature has properties.category)
Run: python pipeline/fetch_known_sites.py
"""
import os, sys, json, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
OVERPASS = "https://overpass-api.de/api/interpreter"
MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]

# Israel bounding box (S, W, N, E). Negev + the whole country.
BBOX = (29.45, 34.20, 33.35, 35.92)

# category -> OSM tag selectors (each a full key/value); 'mask' marks false-positive layers
CATEGORIES = {
    "archaeological_site": dict(tags=[('historic', 'archaeological_site')], mask=False,
                                color="#ff7a3c"),
    "ruins":               dict(tags=[('historic', 'ruins')], mask=False, color="#f5c542"),
    "tell":                dict(tags=[('place', 'locality'), ('historic', 'tell')], mask=False,
                                color="#ffd24a"),
    "industrial":          dict(tags=[('landuse', 'industrial')], mask=True, color="#7aa0ff"),
    "quarry":              dict(tags=[('landuse', 'quarry')], mask=True, color="#9b8cf0"),
    "works":               dict(tags=[('man_made', 'works')], mask=True, color="#6c7bd1"),
}


def overpass(query, tries=3):
    last = None
    for i in range(tries):
        url = MIRRORS[i % len(MIRRORS)]
        try:
            r = requests.post(url, data={"data": query}, timeout=180,
                              headers={"User-Agent": "sherdhunter-research/0.1 (archaeology)"})
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:120]}"
        except Exception as e:
            last = str(e)
        time.sleep(3)
    raise RuntimeError(f"overpass failed: {last}")


def build_query(tags):
    s, w, n, e = BBOX
    sel = "".join(f'["{k}"="{v}"]' for k, v in tags)
    # nodes + ways + relations; 'out center' gives a representative point for areas
    return (f"[out:json][timeout:120];("
            f'node{sel}({s},{w},{n},{e});'
            f'way{sel}({s},{w},{n},{e});'
            f'relation{sel}({s},{w},{n},{e}););out center tags;')


def feature(el, category, color, mask):
    if el["type"] == "node":
        lon, lat = el.get("lon"), el.get("lat")
    else:
        c = el.get("center") or {}
        lon, lat = c.get("lon"), c.get("lat")
    if lon is None or lat is None:
        return None
    t = el.get("tags", {})
    name = t.get("name:en") or t.get("name") or t.get("int_name") or ""
    return {"type": "Feature",
            "properties": {"category": category, "mask": mask, "color": color,
                           "name": name, "name_he": t.get("name", ""),
                           "site_type": t.get("site_type") or t.get("historic") or "",
                           "osm": f'{el["type"]}/{el["id"]}'},
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]}}


def main():
    t0 = time.time()
    feats, counts = [], {}
    seen = set()
    for cat, spec in CATEGORIES.items():
        # tags listed together = AND on one element (e.g. place=locality AND historic=tell);
        # but most categories are a single tag. We issue one query per category.
        q = build_query(spec["tags"])
        try:
            data = overpass(q)
        except RuntimeError as e:
            print(f"  {cat:20s} SKIP ({e})")
            counts[cat] = 0
            continue
        added = 0
        for el in data.get("elements", []):
            key = (el["type"], el["id"])
            if key in seen:
                continue
            f = feature(el, cat, spec["color"], spec["mask"])
            if f:
                feats.append(f)
                seen.add(key)
                added += 1
        counts[cat] = added
        print(f"  {cat:20s} {added:5d}  ({time.time()-t0:.0f}s)")

    out = os.path.join(VIEWER, "known_sites.geojson")
    json.dump({"type": "FeatureCollection",
               "meta": {"source": "OpenStreetMap / Overpass", "bbox": list(BBOX),
                        "counts": counts, "run": time.strftime("%Y-%m-%d %H:%M"),
                        "note": "known-knowledge layer; archaeological=validation, "
                                "industrial/quarry/works=false-positive mask"},
               "features": feats},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    nsite = sum(counts[c] for c in CATEGORIES if not CATEGORIES[c]["mask"])
    nmask = sum(counts[c] for c in CATEGORIES if CATEGORIES[c]["mask"])
    print(f"[{time.time()-t0:.0f}s] wrote {out}: {len(feats)} features "
          f"({nsite} sites, {nmask} mask) -> viewer layer")


if __name__ == "__main__":
    main()
