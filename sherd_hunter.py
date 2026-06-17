"""
SherdHunter v0.1 — Sub-pixel fired-ceramic detection from spaceborne hyperspectral data
========================================================================================
Target:   Tel Arad, Israel (31.2803N, 35.1247E) — EB II lower city surface scatter
Sensor:   NASA EMIT (L2A surface reflectance, ~60m pixels, 285 bands, 380-2500nm)
Physics:  Firing clay above ~550C dehydroxylates kaolinite -> metakaolin.
          The Al-OH absorption at ~2200nm degrades/shifts in a characteristic way,
          making FIRED ceramic spectrally separable from RAW clay and from soil.
Method:   Sub-pixel target detection (ACE / Matched Filter) against a fired-ceramic
          endmember, validated against IAA-mapped site polygons (on-tell vs off-tell ROC).

Pipeline stages:
  1. acquire   — download EMIT granules (requires free NASA Earthdata login)
  2. preprocess— orthorectify via GLT, mask bad bands & deep water-vapor bands
  3. features  — continuum removal over 2050-2350nm (the Al-OH diagnostic window)
  4. endmember — fired-ceramic signature: (a) USGS/ECOSTRESS library brick/ceramic
                 spectra resampled to EMIT bands, AND (b) empirical: mean on-tell
                 pixel spectrum from the densest scatter zone (ground-truth bootstrap)
  5. detect    — ACE (Adaptive Cosine Estimator) + Matched Filter abundance maps
  6. validate  — ROC curve: detector score distribution on-tell vs off-tell background
"""

import numpy as np
import requests

TEL_ARAD = dict(lat=31.2803, lon=35.1247)
BBOX = "35.05,31.23,35.20,31.33"  # lon_min, lat_min, lon_max, lat_max

# Best scenes found 2026-06-11 via CMR query (cloud % is granule-wide; Negev likely clear):
CANDIDATE_GRANULES = [
    "EMIT_L2A_RFL_001_20250616T104758_2516707_014",  # 16% cloud
    "EMIT_L2A_RFL_001_20250616T104810_2516707_015",  # 22% cloud
    "EMIT_L2A_RFL_001_20250723T115748_2520408_025",  # 26% cloud
]

# The diagnostic spectral window (nm)
ALOH_WINDOW = (2050.0, 2350.0)


def search_granules(short_name="EMITL2ARFL", bbox=BBOX, page_size=50):
    """Query NASA CMR (no auth needed for metadata search)."""
    r = requests.get(
        "https://cmr.earthdata.nasa.gov/search/granules.json",
        params={"short_name": short_name, "bounding_box": bbox,
                "page_size": page_size, "sort_key": "-start_date"},
        timeout=30,
    )
    return r.json()["feed"]["entry"]


def download_granule(granule_id, token):
    """Download via earthaccess. Requires NASA Earthdata bearer token (free account).
    >>> import earthaccess; earthaccess.login()  # interactive, or set EARTHDATA_TOKEN
    """
    raise NotImplementedError("Stage 1: plug in Earthdata token, use earthaccess.download()")


def continuum_removal(wavelengths, spectrum, window=ALOH_WINDOW):
    """Convex-hull continuum removal over the Al-OH window.
    Returns depth-normalized spectrum; absorption depth at ~2200nm is the key feature."""
    m = (wavelengths >= window[0]) & (wavelengths <= window[1])
    w, s = wavelengths[m], spectrum[m]
    # Upper convex hull (simple two-anchor version; replace with full hull for production)
    hull = np.interp(w, [w[0], w[-1]], [s[0], s[-1]])
    return w, s / np.clip(hull, 1e-6, None)


def ace_detector(cube, target, mean=None, cov_inv=None):
    """Adaptive Cosine Estimator — the workhorse of sub-pixel target detection.
    cube: (rows, cols, bands), target: (bands,). Returns detection score map."""
    X = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
    if mean is None:
        mean = X.mean(axis=0)
    Xc = X - mean
    t = target - mean
    if cov_inv is None:
        cov = np.cov(Xc, rowvar=False) + 1e-8 * np.eye(Xc.shape[1])
        cov_inv = np.linalg.inv(cov)
    num = (Xc @ cov_inv @ t) ** 2
    den = (t @ cov_inv @ t) * np.einsum("ij,jk,ik->i", Xc, cov_inv, Xc)
    return (num / np.clip(den, 1e-12, None)).reshape(cube.shape[:2])


if __name__ == "__main__":
    granules = search_granules()
    print(f"EMIT granules over Tel Arad: {len(granules)}")
    for g in granules[:5]:
        print(" ", g["time_start"][:10], g["title"])
