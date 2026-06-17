# scenes/

This folder holds **large, reproducible inputs and intermediates** that are intentionally **not**
committed (multi-GB satellite cubes, score grids, the USGS spectral-library zip). The pipeline
writes here.

To populate it:

- **EMIT** (NASA, 60 m): download a granule with a free Earthdata token — see `../SOURCES.md`.
  Example used in this project: `EMIT_L2A_RFL_001_20250616T104810_2516707_015`.
- **EnMAP** (DLR, 30 m): download an L2A scene from the DLR EOC Geoservice (free account) — e.g.
  the Tel Arad / Dead Sea scene `ENMAP01-____L2A-DT0000069168_20240405T091315Z_...` used here.
- **USGS splib07** spectral library: download from the USGS (DOI 10.5066/F7RR1WDJ); the brick /
  mineral endmembers are already extracted into `../pipeline/spectral_library.json`.

Everything in `../viewer/` (the result GeoJSON/PNG/JSON) is committed, so the dashboard works
out-of-the-box without the raw cubes.
