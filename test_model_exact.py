"""
Exactness test: is the shipped transform a member of  H = s * K1 * R * K0^-1 ?

Condition: with N = K1^-1 H K0, we need N N^T = s^2 I.
Unknowns: f0, f1, and both principal points (the destination focal is also
pinned from Adobe's own stored UprightFocalLength35mm in one variant).

If the residual reaches float noise the model is exact; a floor far above
noise means an extra degree of freedom is present.
"""

import os
import re

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1400, 1000
SENSOR = 36.0


def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def to_pix(Hn):
    S = np.diag([1.0 / W, 1.0 / H, 1.0])
    return np.linalg.inv(S) @ Hn @ S


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    o = {}
    for m in re.finditer(r'UprightTransform_(\d)\s*=\s*"([^"]+)"', t):
        v = [float(x) for x in m.group(2).split(",")]
        if len(v) == 9:
            o[int(m.group(1))] = np.array(v).reshape(3, 3)
    o["f35"] = float(re.search(r'UprightFocalLength35mm\s*=\s*([-\d.eE]+)', t).group(1))
    return o


def fit(Hm, f1_fixed, free_src_centre, free_dst_centre, free_f0, free_f1):
    """returns (f0, f1, residual)"""
    idx, p0 = {}, []
    def add(key, val):
        idx[key] = len(p0); p0.append(val)
    if free_f0:
        add("f0", np.log(1600.0))
    if free_f1:
        add("f1", np.log(f1_fixed))
    if free_dst_centre:
        add("cx1", W / 2); add("cy1", H / 2)
    if free_src_centre:
        add("cx0", W / 2); add("cy0", H / 2)

    def unpack(p):
        p = np.asarray(p).ravel()
        g = lambda k, d: (float(np.exp(p[idx[k]])) if k in ("f0", "f1") else float(p[idx[k]])) if idx.get(k) is not None else d
        f0 = g("f0", f1_fixed)
        f1 = g("f1", f1_fixed)
        cx0 = g("cx0", W / 2); cy0 = g("cy0", H / 2)
        cx1 = g("cx1", W / 2); cy1 = g("cy1", H / 2)
        return f0, f1, cx0, cy0, cx1, cy1

    def res(p):
        f0, f1, cx0, cy0, cx1, cy1 = unpack(p)
        N = np.linalg.inv(K_of(f1, cx1, cy1)) @ Hm @ K_of(f0, cx0, cy0)
        M = N @ N.T
        s2 = np.trace(M) / 3.0
        return ((M - s2 * np.eye(3)) / s2).ravel()

    if p0:
        r = least_squares(res, np.array(p0), method="lm", max_nfev=40000)
        p = r.x
    else:
        p = np.array([])
    f0, f1, cx0, cy0, cx1, cy1 = unpack(p)
    N = np.linalg.inv(K_of(f1, cx1, cy1)) @ Hm @ K_of(f0, cx0, cy0)
    M = N @ N.T
    s2 = np.trace(M) / 3.0
    resid = float(np.abs(M - s2 * np.eye(3)).max() / s2)
    Nn = N / np.sqrt(s2)
    return f0, f1, cx0, cy0, cx1, cy1, resid, Nn


def main():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "adobe_truth")) if f.endswith(".lua"))
    table = []
    for fn in files:
        d = parse(os.path.join(HERE, "adobe_truth", fn))
        f35px = d["f35"] / SENSOR * max(W, H)
        for slot in (1, 2, 3, 4):
            if slot not in d:
                continue
            Hm = to_pix(d[slot]); Hm = Hm / Hm[2, 2]
            for label, kw in [
                ("K R K^-1 (shared f)", dict(f1_fixed=f35px, free_src_centre=False,
                                             free_dst_centre=False, free_f0=False, free_f1=False)),
                ("free f0 only", dict(f1_fixed=f35px, free_src_centre=False,
                                      free_dst_centre=False, free_f0=True, free_f1=False)),
                ("free f0 + dst c", dict(f1_fixed=f35px, free_src_centre=False,
                                         free_dst_centre=True, free_f0=True, free_f1=False)),
                ("free f0,f1 + both c", dict(f1_fixed=f35px, free_src_centre=True,
                                             free_dst_centre=True, free_f0=True, free_f1=True)),
            ]:
                f0, f1, cx0, cy0, cx1, cy1, resid, Nn = fit(Hm, **kw)
                ang = np.degrees(np.arccos(np.clip((np.trace(Nn) - 1) / 2, -1, 1)))
                table.append((fn, slot, label, f0, f1, resid, ang))
    print(f"{'file':<20} {'slot':>4} {'model':<20} {'f0':>8} {'f1':>8} {'resid':>10} {'ang':>7}")
    print("-" * 82)
    for r in table:
        print(f"{r[0]:<20} {r[1]:4d} {r[2]:<20} {r[3]:8.1f} {r[4]:8.1f} "
              f"{r[5]:10.2e} {r[6]:7.2f}")
    print("\nmedian residual per model:")
    for m in sorted({r[2] for r in table}):
        e = np.array([r[5] for r in table if r[2] == m])
        print(f"  {m:<22} median {np.median(e):.3e}  max {e.max():.3e}")


if __name__ == "__main__":
    main()
