"""
fetch_geology.py — GSI national geology (1:200k) for the Negev as a known-knowledge MASK layer.

Source: Geological Survey of Israel ArcGIS (open FeatureServer, no auth):
  egozi.gsi.gov.il/arcgis/rest/services/Hosted/Israel_200000_2014_geology/FeatureServer/5
  (layer 5 = GeoFormations polygons; fields symbol/code/name_eng/name_heb)

Why: validate_against_known / composite_enmap show the top composite hotspots are NOT
archaeology but NATURAL bright/fired carbonate terrain. This layer names that terrain so we can
mask it. We tag each formation:
  fired_carbonate  - Hatrurim Fm ("Mottled Zone": natural in-situ combustion -> fired carbonate;
                     the perfect geological false-positive for a fired-ceramic detector)
  lacustrine_marl  - Lisan Fm (Dead Sea Quaternary lake marl; very bright carbonate)
  carbonate        - chalk/marl/limestone (Mount Scopus Grp: Menuha/Mishash/Ghareb, Taqiye, ...)
  evaporite        - Sedom/salt
  alluvium         - Quaternary fill
  other
Israel only (GSI map stops at the border; Jordanian hotspots get no geology — stated limitation).

Out: viewer/geology.geojson  (polygons; properties.flag drives the viewer color + masking)
Run: python pipeline/fetch_geology.py
"""
import os, json, time, re
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
LAYER = ("https://egozi.gsi.gov.il/arcgis/rest/services/Hosted/"
         "Israel_200000_2014_geology/FeatureServer/5/query")
BBOX = (34.55, 30.00, 36.00, 31.75)            # W,S,E,N — Negev + Dead Sea + Arad/Tel Arad

FLAG = {
    "fired_carbonate": (re.compile(r"hatrurim|mottled", re.I), "#ff4d4d"),
    "lacustrine_marl": (re.compile(r"lisan", re.I), "#ff9e3d"),
    "evaporite":       (re.compile(r"sedom|sdom|halite|salt|evaporit", re.I), "#d98cff"),
    "carbonate":       (re.compile(r"chalk|marl|limestone|dolomit|menuha|mishash|ghareb|"
                                   r"taqiye|mount scopus|nezer|zafit|avedat|bina|nezer", re.I),
                        "#7ad1ff"),
    "alluvium":        (re.compile(r"alluvi|fluvial|gravel|sand|loess|playa|conglomerat", re.I),
                        "#cdb892"),
}
DEFAULT_COLOR = "#9aa7b2"


def classify(name_eng, name_heb):
    s = f"{name_eng} {name_heb}"
    for flag, (rx, _col) in FLAG.items():
        if rx.search(s):
            return flag
    return "other"


def color_of(flag):
    return FLAG[flag][1] if flag in FLAG else DEFAULT_COLOR


def esri_rings_to_geojson(rings):
    """esri polygon rings -> GeoJSON Polygon coords (outer+holes passed through; winding
    ignored by Leaflet). Round coords to 5 dp."""
    return [[[round(x, 5), round(y, 5)] for x, y in ring] for ring in rings]


def fetch():
    w, s, e, n = BBOX
    params = {
        "where": "1=1",
        "geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outFields": "symbol,code,name_eng,name_heb",
        "returnGeometry": "true", "maxAllowableOffset": "0.002",   # ~200 m generalization
        "geometryPrecision": "5", "f": "json", "resultRecordCount": "1000",
    }
    feats, offset = [], 0
    while True:
        params["resultOffset"] = offset
        r = requests.get(LAYER, params=params, timeout=120,
                         headers={"User-Agent": "sherdhunter-research/0.1"})
        j = r.json()
        chunk = j.get("features", [])
        for f in chunk:
            g = f.get("geometry") or {}
            rings = g.get("rings")
            if not rings:
                continue
            a = f.get("attributes", {})
            flag = classify(a.get("name_eng", ""), a.get("name_heb", ""))
            feats.append({"type": "Feature",
                          "properties": {"name": a.get("name_eng", ""), "name_he": a.get("name_heb", ""),
                                         "symbol": a.get("symbol", ""), "flag": flag,
                                         "color": color_of(flag)},
                          "geometry": {"type": "Polygon", "coordinates": esri_rings_to_geojson(rings)}})
        print(f"  offset {offset}: +{len(chunk)} (total {len(feats)})")
        if not j.get("exceededTransferLimit") or not chunk:
            break
        offset += len(chunk)
    return feats


def main():
    t0 = time.time()
    feats = fetch()
    counts = {}
    for f in feats:
        fl = f["properties"]["flag"]
        counts[fl] = counts.get(fl, 0) + 1
    out = os.path.join(VIEWER, "geology.geojson")
    json.dump({"type": "FeatureCollection",
               "meta": {"source": "Geological Survey of Israel 1:200,000 (2014), FeatureServer",
                        "bbox": list(BBOX), "counts": counts, "run": time.strftime("%Y-%m-%d %H:%M"),
                        "note": "flag=fired_carbonate(Hatrurim)/lacustrine_marl(Lisan)/carbonate/"
                                "evaporite/alluvium = natural false-positive terrain for the index"},
               "features": feats},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[{time.time()-t0:.0f}s] wrote {out}: {len(feats)} polygons  flags={counts}")


if __name__ == "__main__":
    main()
