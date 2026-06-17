"""
enmap_fetch.py — search + windowed-download EnMAP L2A over an AOI via DLR STAC + EOC SSO.

EnMAP downloads sit behind DLR's CAS single sign-on (sso.eoc.dlr.de). With an EOC account
(free), this script logs in once, keeps the session cookie, and then reads ONLY the AOI
window from the Cloud-Optimized GeoTIFF assets — no multi-GB scene downloads.

Usage:
  set EOC_USER=...            (or pass --user)
  set EOC_PASS=...            (or it prompts; NEVER store credentials in the repo)
  python pipeline/enmap_fetch.py --bbox 35.05,31.23,35.20,31.33 --max-cloud 10

Outputs: scenes/enmap_<sceneid>_aoi.npz  (int16 cube bands x h x w, transform, crs, wl)
"""
import os, sys, re, json, time, getpass
import numpy as np
import requests

STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A/items"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")


def stac_search(bbox, max_cloud=10, limit=100):
    r = requests.get(STAC, params={"bbox": ",".join(map(str, bbox)), "limit": limit}, timeout=60)
    feats = r.json().get("features", [])
    def cc(f):
        try:
            return float(f["properties"].get("eo:cloud_cover", 100))
        except (TypeError, ValueError):
            return 100.0
    good = [f for f in feats if cc(f) <= max_cloud]
    good.sort(key=cc)
    return good


def cas_login(session, service_url, user, pwd, idp="eolab2_enmap-dl"):
    """DLR EOC CAS login. Accounts live in TWO different stores:
      - direct CAS accounts (registered at sso.eoc.dlr.de) -> plain username/password POST
      - EO-LAB accounts (registered at eo-lab.org)         -> DELEGATED auth: CAS redirects to
        the EO-Lab Keycloak (auth.fra1-1.cloudferro.com, realm eo-lab), we authenticate
        THERE, and Keycloak bounces us back to CAS with an OIDC code -> service ticket.
    Tries the delegated EO-LAB route first (that's where EnMAP downloads are provisioned,
    client_name=eolab2_enmap-dl), falls back to the direct CAS form.
    IMPORTANT: call with a SMALL service_url (e.g. the metadata XML) — the final redirect
    lands on the service and we read its body."""
    login_url = "https://sso.eoc.dlr.de/eoc/auth/login"
    r = session.get(login_url, params={"service": service_url}, timeout=60)
    if 'name="execution"' not in r.text:
        return r          # no form -> SSO session already active, we were passed through

    if f'value="{idp}"' in r.text:
        execution = re.search(r'name="execution" value="([^"]+)"', r.text).group(1)
        csrf = re.search(r'name="_csrf" value="([^"]+)"', r.text)
        data = {"execution": execution, "_eventId": "delegatedAuthenticationRedirect",
                "client_name": idp}
        if csrf:
            data["_csrf"] = csrf.group(1)
        r1 = session.post(login_url, params={"service": service_url},
                          data=data, timeout=120, allow_redirects=True)
        act = re.search(r'<form[^>]+action="([^"]+)"', r1.text)
        if act and "login-actions" in act.group(1):
            action = act.group(1).replace("&amp;", "&")
            r2 = session.post(action, data={"username": user, "password": pwd,
                                            "credentialId": "", "login": "Sign In"},
                              timeout=120, allow_redirects=True)
            if "login-actions" in getattr(r2, "url", "") or 'name="password"' in r2.text:
                msg = re.search(r'(?:kc-feedback-text|input-error|alert-error)[^>]*>([^<]+)',
                                r2.text)
                detail = msg.group(1).strip()[:200] if msg else "(no message)"
                sys.exit(f"EO-LAB (Keycloak) login failed for '{user}': {detail}")
            print("EO-LAB delegated login OK")
            return r2
        print("EO-LAB delegation did not reach a Keycloak form; trying direct CAS login")
        r = session.get(login_url, params={"service": service_url}, timeout=60)

    # direct CAS credentials (old-style EOC accounts)
    fields = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', r.text))
    fields.update({"username": user, "password": pwd,
                   "_eventId": "submit", "geolocation": "", "deviceFingerprint": ""})
    r2 = session.post(login_url, params={"service": service_url},
                      data=fields, timeout=120, allow_redirects=True)
    if r2.status_code >= 400 or 'name="execution"' in r2.text:
        msg = re.search(r'class="[^"]*alert[^"]*"[^>]*>(.*?)</div>', r2.text, re.S)
        detail = re.sub(r"<[^>]+>", " ", msg.group(1)).strip()[:220] if msg else "(no message)"
        sys.exit(f"EOC SSO login failed (status {r2.status_code}): {detail}")
    return r2


def fetch_aoi(asset_url, bbox, session):
    """Windowed read of a COG through the authenticated session.
    GDAL reads via /vsicurl with the session cookies exported to a cookie header."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    cookies = "; ".join(f"{c.name}={c.value}" for c in session.cookies)
    env = rasterio.Env(GDAL_HTTP_HEADERS=f"Cookie: {cookies}",
                       GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF")
    with env:
        with rasterio.open("/vsicurl/" + asset_url) as ds:
            wb = transform_bounds("EPSG:4326", ds.crs, *bbox)
            win = from_bounds(*wb, ds.transform).round_offsets().round_lengths()
            cube = ds.read(window=win)
            return cube, np.array(ds.window_transform(win))[:6], str(ds.crs), ds.nodata


def wavelengths_from_meta(meta_url, session):
    """Parse band center wavelengths from the scene METADATA.XML."""
    xml = session.get(meta_url, timeout=60).text
    wl = [float(x) for x in re.findall(r"<wavelengthCenterOfBand>([\d.]+)</wavelengthCenterOfBand>", xml)]
    return np.array(wl, dtype=np.float64)


def main():
    args = sys.argv[1:]
    bbox = tuple(float(x) for x in (args[args.index("--bbox")+1].split(",")
                 if "--bbox" in args else "35.05,31.23,35.20,31.33".split(",")))
    max_cloud = float(args[args.index("--max-cloud")+1]) if "--max-cloud" in args else 10
    user = (args[args.index("--user")+1] if "--user" in args else os.environ.get("EOC_USER")) \
           or input("EOC username: ")
    pwd = os.environ.get("EOC_PASS") or getpass.getpass("EOC password: ")

    feats = stac_search(bbox, max_cloud)
    if not feats:
        sys.exit("no EnMAP scenes under cloud threshold for this bbox")
    f = feats[0]
    sid = f["id"]
    print("best scene:", sid, "| cloud:", f["properties"].get("eo:cloud_cover"),
          "|", f["properties"].get("datetime", "")[:10])
    img = f["assets"]["image"]["href"]
    meta = f["assets"]["metadata"]["href"]

    s = requests.Session()
    s.headers["User-Agent"] = "sherdhunter-research/0.1"
    # login against the small metadata XML (its body doubles as the wavelength source)
    r = cas_login(s, meta, user, pwd)
    xml = r.text if "<wavelengthCenterOfBand>" in r.text else s.get(meta, timeout=60).text
    wl = np.array([float(x) for x in
                   re.findall(r"<wavelengthCenterOfBand>([\d.]+)</wavelengthCenterOfBand>", xml)],
                  dtype=np.float64)
    print(f"SSO login OK; {len(wl)} band centers parsed from metadata")

    t0 = time.time()
    os.makedirs(SCENES, exist_ok=True)
    try:
        cube, transform, crs, nodata = fetch_aoi(img, bbox, s)
        print(f"[{time.time()-t0:.1f}s] windowed COG read OK")
    except Exception as e:
        # fallback: stream the full TIF locally, then window it
        print(f"windowed read failed ({e}); falling back to full download ...")
        local = os.path.join(SCENES, sid[:38] + "_SPECTRAL.TIF")
        if not os.path.exists(local):
            with s.get(img, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(local, "wb") as fh:
                    for chunk in resp.iter_content(1 << 22):
                        fh.write(chunk)
            print(f"[{time.time()-t0:.1f}s] downloaded {os.path.getsize(local)/1e9:.2f} GB")
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        with rasterio.open(local) as ds:
            wb = transform_bounds("EPSG:4326", ds.crs, *bbox)
            win = from_bounds(*wb, ds.transform).round_offsets().round_lengths()
            cube = ds.read(window=win)
            transform, crs, nodata = np.array(ds.window_transform(win))[:6], str(ds.crs), ds.nodata

    out = os.path.join(SCENES, f"enmap_{sid[:38]}_aoi.npz")
    np.savez_compressed(out, cube=cube.astype(np.int16), transform=transform,
                        crs=crs, nodata=nodata if nodata is not None else -32768,
                        wl=wl, bbox=np.array(bbox))
    print(f"[{time.time()-t0:.1f}s] saved {out}  cube {cube.shape}  wl {len(wl)} bands")


if __name__ == "__main__":
    main()
