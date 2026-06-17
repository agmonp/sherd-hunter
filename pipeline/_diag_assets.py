"""Print full asset URLs for the best EnMAP scene (for a browser-session download)."""
import requests

STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A/items"

def cc(f):
    try:
        return float(f["properties"].get("eo:cloud_cover", 100))
    except (TypeError, ValueError):
        return 100.0

r = requests.get(STAC, params={"bbox": "35.05,31.23,35.20,31.33", "limit": 100}, timeout=60)
feats = [f for f in r.json()["features"] if cc(f) <= 10]
feats.sort(key=cc)
f = feats[0]
print("scene:", f["id"])
print("cloud:", f["properties"].get("eo:cloud_cover"), "| date:", f["properties"].get("datetime", "")[:10])
for name, a in f["assets"].items():
    print(f"\n[{name}] {a.get('title','')}")
    print(" ", a["href"])
