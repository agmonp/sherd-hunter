# Open hyperspectral data sources for fired-ceramic detection

The detection tiers, from free orbital screening down to cm-scale confirmation. **Resolution sets
what is possible.** At 30–60 m the realistic win is *statistical separation* (ROC, on-tell vs
off-tell) and *anomaly flagging* — not a sherd map. Pottery **typology** (ware/fabric classes) only
becomes feasible at the airborne/drone/lab tier (≤1 m → cm).

| Sensor | Res | Bands / range | Access | Role in this project |
|---|---|---|---|---|
| **EMIT** (NASA/JPL, ISS) | 60 m | 285 · 0.38–2.50 µm | Free, Earthdata token | **Current screening sensor.** Built for surface mineralogy → strong on clay/carbonate. |
| **EnMAP** (DLR) | 30 m | 224 · 0.42–2.45 µm | Free · **taskable** via science proposal | **The key next step** — 4× EMIT pixel area, can be pointed at a target. |
| **PRISMA** (ASI) | 30 m | 239 · 0.40–2.50 µm | Free, registration | Alt archive; growing coverage over the Levant. |
| **Hyperion** (EO-1, USGS) | 30 m | 242 · 0.40–2.50 µm | Free, EarthExplorer | 2001–2017 archive → **temporal stacking** to boost SNR on a static target. |
| **DESIS** (DLR, ISS) | 30 m | 235 · 0.40–1.00 µm | Free, EOWEB | VNIR-only (no SWIR) → weak for the 2200/2345 nm diagnostics. Context only. |
| **Sentinel-2 / Landsat 8-9** | 10–30 m | multispectral | Free | Not hyperspectral, but dense time series — crop/soil marks, change detection, context. |
| **Pixxel Fireflies** | ~5 m | hyperspectral | Commercial / research access | Candidate interrogation once an anomaly is flagged. |
| **TAU Remote Sensing Lab** (Ben-Dor) | ~1 m | airborne AisaFENIX | Academic collaboration | Israeli campaigns + national soil spectral library. **Validation tier.** |
| **Survey of Israel orthophoto** | 12–25 cm | RGB | Free viewer (govmap) | Eyeball anomalies before committing field time. |
| **Drone HSI** (Specim / Resonon) | cm | hyperspectral | Rental / collaboration | **Where pottery-fabric typology becomes feasible.** Final confirmation. |
| SHALOM (ISA/ASI) | ~10 m | hyperspectral | **Not launched** (in development) | Future Israeli option. |

## Spectral reference libraries (for non-circular endmembers + future classification)

| Library | What | Access | Role |
|---|---|---|---|
| **USGS Spectral Library v7** | ~2,400 lab spectra: minerals, soils, man-made | Free, no login — DOI 10.5066/F7RR1WDJ (5.5 GB zip; ranges not supported) | **Fired-ceramic / clay / calcite endmembers**; geology discrimination |
| **ECOSTRESS library** (JPL) | incl. man-made brick/tile spectra | Free, registration form | Alternative ceramic endmembers |
| openhsi (github.com/openhsi/openhsi) | open-source ~$3–5k pushbroom HSI camera + Python stack | open source | **Field tier someday**: cheap drone/ground camera for cm-scale confirmation — not needed for orbital work |
| PlantSpecLab | plant spectroscopy tooling | open source | marginal here (vegetation is sparse at our sites) |

## Programmatic EnMAP access (the good path — found 2026-06-12)

DLR's **STAC API** indexes the EnMAP archive — searchable with NO auth:
`https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A/items?bbox=...`
**28 archive scenes cover the Tel Arad box; several 0% cloud (2024-04-05, 2024-01-22, 2023-11-29).**
Assets are Cloud-Optimized GeoTIFFs (`SPECTRAL_IMAGE_COG.TIF`, 224 bands, 30 m) + quality masks —
windowed AOI reads possible (no full-scene downloads). Downloads sit behind **EOC CAS SSO**
(sso.eoc.dlr.de) — needs the (free) EOC account; `pipeline/enmap_fetch.py` does login + AOI fetch.
EOWEB GeoPortal is only needed for *tasking* proposals, not archive pulls.

## Access notes

**EMIT (use now)** — Metadata search needs *no* login (NASA CMR; the viewer's "Scan" tab and
`sherd_hunter.search_granules()` both hit it). Downloading the ~1.8–3.4 GB granules needs a free
[Earthdata](https://urs.earthdata.nasa.gov/) account. Get a bearer token, then:
```
setx EARTHDATA_TOKEN "<token>"          # rotates ~60 days — never commit it
python pipeline/detect.py EMIT_L2A_RFL_001_20250616T104810_2516707_015
```
EMIT L2A `.nc` layout: root `reflectance` (downtrack, crosstrack, bands=285); group `location`
(lat/lon + GLT for orthorectification); group `sensor_band_parameters` (wavelengths 381–2493 nm).
Mask fill (−9999 / −0.01) and deep water-vapor bands (~1340–1465, ~1790–1960 nm) before any math.

**EnMAP (next)** — Free but request-based. Either pull archived scenes or submit a **tasking /
science proposal** ("fired-ceramic detection validation over Negev tells") at
<https://planning.enmap.org/>. 30 m ≈ 4× the on-target signal fraction vs EMIT.

**PRISMA** — Register at <https://prisma.asi.it/>, then browse/order by AOI.

**Hyperion** — <https://earthexplorer.usgs.gov/> → dataset "EO-1 Hyperion". Many scenes over Israel
across 2001–2017; co-register and average to raise SNR on a static scatter.

**TAU (Eyal Ben-Dor's lab)** — Airborne hyperspectral campaigns over Israel + a national soil
spectral library; he is a SHALOM mission author. The natural academic partner for ground-truth and
the cm-scale tier. Worth an outreach email **once the ROC is convincing** — not before.

## The physics these sources feed
- Clay minerals: diagnostic **Al-OH absorption ~2200 nm** (SWIR).
- Firing >~550 °C dehydroxylates kaolinite → metakaolin: the 2200 nm feature **degrades** —
  fired ceramic separates from raw clay and soil. *(First Tel Arad pixel: on-tell Al-OH depth
  0.0164 vs background 0.0289 — shallower on-tell, the predicted direction.)*
- Negev terrain is carbonate (chalk/limestone): dominant feature **~2345 nm** (calcite).
  Anthropogenic ash / lime-plaster can **deepen** it → an anthrosol halo. *(First pixel: on-tell
  0.0675 vs 0.0457, ~48% deeper.)*
- **Caution:** natural carbonate/clay outcrops produce the same features. Test every detector
  against off-site outcrops before calling a signal anthropogenic.
