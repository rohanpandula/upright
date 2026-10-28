"""
Constrained test of the model H = s * K(f1,c1) * R * K(f0,c0)^-1 using the
parameters Adobe itself stored (UprightFocalLength35mm, UprightCenterNormX/Y)
for the destination intrinsics.

If the shipped model is exactly this, the fit residual collapses to float noise
once the destination intrinsics are pinned to Adobe's own stored values and only
(s, R, f0) are free.
"""

import json
import os
import re

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.join(HERE, "adobe_truth")
W, H = 1400, 1000
SENSOR = 36.0


def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def to_pix(Hn):
    S = np.diag([1.0 / W, 1.0 / H, 1.0])
    return np.linalg.inv(S) @ Hn @ S


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r'UprightTransform_(\d)\s*=\s*"([^"]+)"', t):
        v = [float(x) for x in m.group(2).split(",")]
        if len(v) == 9:
            out[int(m.group(1))] = np.array(v).reshape(3, 3)
    g = lambda k: float(re.search(k + r'\s*=\s*([-\d.eE]+)', t).group(1))  # noqa
    out["f35"] = g("UprightFocalLength35mm")
    out["cx"] = g("UprightCenterNormX")
    out["cy"] = g("UprightCenterNormY")
    fx = re.search(r'PerspectiveHorizontal\s*=\s*([-\d.eE]+)', t)
    out["ph"] = float(fx.group(1)) if fx else 0.0
    return out


def build(p, f1, cx1, cy1, free_centre):
    s = np.exp(p[0])
    R = rot(p[1], p[2], p[3])
    f0 = np.exp(p[4])
    if free_centre:
        cx0, cy0 = p[5], p[6]
    else:
        cx0, cy0 = W / 2, H / 2
    K1 = K_of(f1, cx1, cy1)
    K0 = K_of(f0, cx0, cy0)
    return s * K1 @ R @ np.linalg.inv(K0)


def run(free_centre):
    files = sorted(f for f in os.listdir(TRUTH_DIR) if f.endswith(".lua"))
    rows = []
    for fn in files:
        d = parse(os.path.join(TRUTH_DIR, fn))
        f1 = d["f35"] / SENSOR * max(W, H)
        cx1, cy1 = d["cx"] * W, d["cy"] * H
        for slot in (1, 2, 4):
            if slot not in d:
                continue
            Ht = to_pix(d[slot])
            Ht = Ht / Ht[2, 2]
            p0 = np.array([0, .1, .05, 0, np.log(1700.0)]
                          + ([W / 2, H / 2] if free_centre else []))

            def res(p):
                Hp = build(p, f1, cx1, cy1, free_centre)
                Hp = Hp / Hp[2, 2]
                return (Hp - Ht).ravel() * np.array([1, 1, 1, 1, 1, 1, .3, .3, 1.])

            r = least_squares(res, p0, method="lm", max_nfev=40000)
            Hp = build(r.x, f1, cx1, cy1, free_centre)
            Hp = Hp / Hp[2, 2]
            err = np.abs(Hp - Ht).max()
            rows.append(dict(file=fn, slot=slot, maxerr=err,
                             f0=np.exp(r.x[4]), f1=f1, scale=np.exp(r.x[0]),
                             rx=np.degrees(r.x[1]), ry=np.degrees(r.x[2]),
                             rz=np.degrees(r.x[3])))
    return rows


def main():
    for free_centre in (False, True):
        rows = run(free_centre)
        e = np.array([r["maxerr"] for r in rows])
        tag = "freeK0centre" if free_centre else "K0=image-centre"
        print(f"\n=== model {tag}: {len(rows)} fits, "
              f"median maxerr {np.median(e):.3e}, max {e.max():.3e} ===")
        print(f"{'file':<20} {'slot':>4} {'maxerr':>10} {'f0':>8} {'f1':>8} "
              f"{'rx':>8} {'ry':>8} {'rz':>8}")
        for r in rows:
            print(f"{r['file']:<20} {r['slot']:4d} {r['maxerr']:10.2e} "
                  f"{r['f0']:8.1f} {r['f1']:8.1f} {r['rx']:8.3f} "
                  f"{r['ry']:8.3f} {r['rz']:8.3f}")
        with open(os.path.join(HERE, f"adobe_constrained_{'free' if free_centre else 'fixed'}.json"), "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
