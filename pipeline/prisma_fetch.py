"""
prisma_fetch.py — read a PRISMA L2D HDF5 (.he5) into a reflectance cube for the SherdHunter
pipeline (cross-sensor coverage + the pre-registered EnMAP<->PRISMA agreement test).

STATUS: UNTESTED — written from the PRISMA Product Specification, to be verified/fixed against
the FIRST real scene. Run it on the .he5 and it first DUMPS the structure (groups, datasets,
shapes, root attributes) so the exact field names/scaling can be confirmed, THEN best-effort
extracts the cube. PRISMA access is manual: register at prismauserregistration.asi.it (research
account; institutional email preferred), search/order at prisma.asi.it, download L2D .he5.
Needs: pip install h5py   (not yet installed in this env).

Documented L2D layout (verify on first file):
  root attrs: List_Cw_Vnir (66), List_Cw_Swir (173)  [some entries 0 = unused bands];
              L2ScaleVnirMin/Max, L2ScaleSwirMin/Max  (DN->refl = min + DN*(max-min)/65535);
              Epsg_Code, Product_ULcorner_easting/northing, pixel size 30 m.
  cubes: /HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube (rows,66,cols), SWIR_Cube (rows,173,cols)
Out: scenes/prisma_<id>_aoi.npz (cube[H,W,bands], wl, transform, crs)
Run: python pipeline/prisma_fetch.py <file.he5>
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")


def dump(h5):
    print("=== HDF5 structure ===")
    def visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset):
            print(f"  DSET {name}  shape={obj.shape} dtype={obj.dtype}")
    h5.visititems(visit)
    print("=== root attrs (key ones) ===")
    for k in h5.attrs:
        if any(s in k for s in ("Cw", "Scale", "Epsg", "corner", "Pixel", "Fwhm", "Flag")):
            v = h5.attrs[k]
            print(f"  {k} = {np.array(v).ravel()[:6]}{'...' if np.size(v) > 6 else ''}")


def main(path):
    import h5py
    f = h5py.File(path, "r")
    dump(f)

    def find_cube(substr):
        hit = []
        f.visititems(lambda n, o: hit.append(n) if (hasattr(o, "shape") and substr in n) else None)
        return hit[0] if hit else None

    try:
        vname, sname = find_cube("VNIR_Cube"), find_cube("SWIR_Cube")
        vnir = np.array(f[vname]); swir = np.array(f[sname])           # (rows, bands, cols)
        cw_v = np.array(f.attrs["List_Cw_Vnir"], float)
        cw_s = np.array(f.attrs["List_Cw_Swir"], float)
        smin_v, smax_v = float(f.attrs["L2ScaleVnirMin"]), float(f.attrs["L2ScaleVnirMax"])
        smin_s, smax_s = float(f.attrs["L2ScaleSwirMin"]), float(f.attrs["L2ScaleSwirMax"])
        # DN -> reflectance; move band axis last -> (rows, cols, bands)
        vnir = (smin_v + vnir.astype(np.float32) * (smax_v - smin_v) / 65535.0).transpose(0, 2, 1)
        swir = (smin_s + swir.astype(np.float32) * (smax_s - smin_s) / 65535.0).transpose(0, 2, 1)
        cube = np.concatenate([vnir, swir], axis=2)
        wl = np.concatenate([cw_v, cw_s])
        good = wl > 0                                                  # drop unused (Cw==0) bands
        cube, wl = cube[:, :, good], wl[good]
        order = np.argsort(wl); cube, wl = cube[:, :, order], wl[order]
        H, W = cube.shape[:2]

        epsg = int(np.array(f.attrs["Epsg_Code"]).ravel()[0])
        ulx = float(np.array(f.attrs["Product_ULcorner_easting"]).ravel()[0])
        uly = float(np.array(f.attrs["Product_ULcorner_northing"]).ravel()[0])
        px = 30.0
        transform = np.array([ulx, px, 0.0, uly, 0.0, -px])
        out = os.path.join(SCENES, os.path.basename(path).replace(".he5", "") + "_cube.npz")
        np.savez_compressed(out, cube=cube.astype(np.float32), wl=wl,
                            transform=transform, crs=f"EPSG:{epsg}")
        print(f"OK -> {out}  cube {cube.shape}  wl {wl.min():.0f}-{wl.max():.0f} nm  EPSG:{epsg}")
    except Exception as e:
        print(f"\nEXTRACTION FAILED ({e}). Use the structure dump above to fix field names/attrs.")
    f.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python pipeline/prisma_fetch.py <PRS_L2D_*.he5>   (needs: pip install h5py)")
    main(sys.argv[1])
