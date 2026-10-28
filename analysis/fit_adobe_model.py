"""
Recover the exact algebraic model Adobe's Upright solver uses, from the
transforms the real Lightroom Classic wrote into the catalog.

Candidate models (all in pixel coordinates, H maps source -> destination):

  sharedK : H = s * K(f) R K(f)^-1
  freeK   : H = s * K(f1, c1) R K(f0, c0)^-1        (asymmetric intrinsics)
  freeKc  : freeK with the destination principal point free and the source
            principal point fixed at the image centre

A homography has 8 DOF. `sharedK` has 5 free parameters (s, f, rx, ry, rz) and
`freeK` has 8-10. If a model fits to ~1e-9 the model is *exactly* what shipped;
a residual floor well above float noise proves the model is wrong.
"""

import json
import os
import re
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.join(HERE, "adobe_truth")
W, H = 1400, 1000


def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def to_pix(Hn, w, h):
    S = np.diag([1.0 / w, 1.0 / h, 1.0])
    return np.linalg.inv(S) @ Hn @ S


def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r'UprightTransform_(\d)\s*=\s*"([^"]+)"', txt):
        v = [float(x) for x in m.group(2).split(",")]
        if len(v) == 9:
            out[int(m.group(1))] = np.array(v).reshape(3, 3)
    cen = re.search(r'UprightCenterNormX\s*=\s*([-\d.eE]+)', txt)
    ceny = re.search(r'UprightCenterNormY\s*=\s*([-\d.eE]+)', txt)
    foc = re.search(r'UprightFocalLength35mm\s*=\s*([-\d.eE]+)', txt)
    meta = re.search(r'UprightFocalMode\s*=\s*([-\d.eE]+)', txt)
    dig = re.search(r'UprightDependentDigest\s*=\s*"([^"]*)"', txt)
    out["_meta"] = dict(
        cx=float(cen.group(1)) if cen else None,
        cy=float(ceny.group(1)) if ceny else None,
        f35=float(foc.group(1)) if foc else None,
        fmode=meta.group(1) if meta else None,
        digest=dig.group(1) if dig else None)
    return out


def unpack(p, model):
    """-> (s, f1, cx1, cy1, f0, cx0, cy0, R)"""
    s = np.exp(p[0])
    R = rot(p[1], p[2], p[3])
    if model == "sharedK":
        f = np.exp(p[4])
        return s, f, W / 2, H / 2, f, W / 2, H / 2, R
    if model == "freeK":
        f0, f1 = np.exp(p[4]), np.exp(p[5])
        return s, f1, W / 2, H / 2, f0, W / 2, H / 2, R
    if model == "freeKc":
        f0, f1, cx1, cy1 = np.exp(p[4]), np.exp(p[5]), p[6], p[7]
        return s, f1, cx1, cy1, f0, W / 2, H / 2, R
    raise ValueError(model)


def build(p, model):
    s, f1, cx1, cy1, f0, cx0, cy0, R = unpack(p, model)
    return s * K_of(f1, cx1, cy1) @ R @ np.linalg.inv(K_of(f0, cx0, cy0))


def resid(p, Ht, model):
    Hp = build(p, model)
    Hp = Hp / Hp[2, 2]
    Hn = Ht / Ht[2, 2]
    r = (Hp - Hn).ravel()
    # weight the bottom row less (it is fixed by normalisation)
    w = np.array([1, 1, 1, 1, 1, 1, .3, .3, 1.0])
    return r * w


def fit(Ht, model, seed=1600.0):
    best = None
    for f0 in (seed * 0.6, seed, seed * 1.6):
        if model == "sharedK":
            p0 = np.array([0, 0.2, 0.1, 0.0, np.log(f0)])
        elif model == "freeK":
            p0 = np.array([0, 0.2, 0.1, 0.0, np.log(f0), np.log(f0)])
        else:
            p0 = np.array([0, 0.2, 0.1, 0.0, np.log(f0), np.log(f0), W / 2, H / 2])
        try:
            r = least_squares(resid, p0, args=(Ht, model), method="lm", max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def main():
    files = sorted(f for f in os.listdir(TRUTH_DIR) if f.endswith(".lua"))
    print(f"catalog records: {len(files)}")
    rows = []
    for fn in files:
        d = parse(os.path.join(TRUTH_DIR, fn))
        for slot, Hn in d.items():
            if slot == "_meta":
                continue
            Ht = to_pix(Hn, W, H)
            for model in ("sharedK", "freeK", "freeKc"):
                r = fit(Ht, model)
                if r is None:
                    continue
                Hp = build(r.x, model)
                Hp = Hp / Hp[2, 2]
                err = np.abs(Hp - Ht / Ht[2, 2]).max()
                s, f1, cx1, cy1, f0, cx0, cy0, R = unpack(r.x, model)
                ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                rows.append(dict(file=fn, slot=slot, model=model, maxerr=err,
                                 f0=f0, f1=f1, cx1=cx1, cy1=cy1,
                                 rx=np.degrees(r.x[1]), ry=np.degrees(r.x[2]),
                                 rz=np.degrees(r.x[3]), rot_deg=ang,
                                 scale=s))
    print(f"\n{'model':<9} {'n':>3} {'maxerr med':>12} {'maxerr max':>12} "
          f"{'f0 med':>9} {'f1 med':>9} {'|R| med':>8}")
    for model in ("sharedK", "freeK", "freeKc"):
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        e = np.array([r["maxerr"] for r in sub])
        f0 = np.array([r["f0"] for r in sub])
        f1 = np.array([r["f1"] for r in sub])
        rg = np.array([r["rot_deg"] for r in sub])
        print(f"{model:<9} {len(sub):3d} {np.median(e):12.3e} {e.max():12.3e} "
              f"{np.median(f0):9.1f} {np.median(f1):9.1f} {np.median(rg):8.3f}")
    with open(os.path.join(HERE, "adobe_model_fit.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\nper-slot detail (freeKc):")
    print(f"{'file':<22} {'slot':>4} {'maxerr':>10} {'f0':>8} {'f1':>8} "
          f"{'rx':>8} {'ry':>8} {'rz':>8}")
    for r in rows:
        if r["model"] != "freeKc":
            continue
        print(f"{r['file']:<22} {r['slot']:4d} {r['maxerr']:10.2e} {r['f0']:8.1f} "
              f"{r['f1']:8.1f} {r['rx']:8.3f} {r['ry']:8.3f} {r['rz']:8.3f}")


if __name__ == "__main__":
    main()
