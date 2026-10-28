"""
Deterministic test of the model  H = s * K1 * R * K0^-1 .

No optimiser guesswork. If H = s K1 R K0^-1 with K's upper-triangular and
R a rotation then

    N = K1^-1 H K0     must satisfy     N N^T = s^2 I

That is 6 scalar equations. We know the destination focal from Adobe's own
UprightFocalLength35mm, so the remaining unknowns are only f0 (source focal)
and the destination principal point. Solving them makes N N^T / s^2 -> I with a
residual that reveals whether the model is exactly right.

We also test the competing hypotheses:
  * source focal == destination focal (pure rotation + zoom, "sharedK")
  * destination principal point == image centre
"""

import os
import re

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.join(HERE, "adobe_truth")
W, H = 1400, 1000
SENSOR = 36.0


def K_of(f, cx, cy):
    f = float(np.asarray(f).ravel()[0]); cx = float(cx); cy = float(cy)
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


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
    return out


def orth_resid(Hm, f0, f1, cx0, cy0, cx1, cy1):
    N = np.linalg.inv(K_of(f1, cx1, cy1)) @ Hm @ K_of(f0, cx0, cy0)
    M = N @ N.T
    s2 = np.trace(M) / 3.0
    return float(np.abs(M - s2 * np.eye(3)).max() / s2)


def solve(Hm, f1, cx1, cy1, free_centre, free_f0):
    """minimise the orthogonality residual over the remaining intrinsics."""
    def pack(p):
        p = np.asarray(p).ravel()
        i = 0
        f0 = np.exp(p[i]) if free_f0 else f1
        i += 1 if free_f0 else 0
        if free_centre:
            c1x, c1y = p[i], p[i + 1]
        else:
            c1x, c1y = cx1, cy1
        return f0, c1x, c1y

    p0 = ([np.log(f1)] if free_f0 else []) + ([cx1, cy1] if free_centre else [])

    def res(p):
        f0, c1x, c1y = pack(p)
        N = np.linalg.inv(K_of(f1, c1x, c1y)) @ Hm @ K_of(f0, W / 2, H / 2)
        M = N @ N.T
        s2 = np.trace(M) / 3.0
        return ((M - s2 * np.eye(3)) / s2).ravel()

    r = least_squares(res, np.array(p0) if len(p0) else np.array([0.0]),
                      method="lm", max_nfev=20000)
    f0, c1x, c1y = pack(r.x) if len(p0) else (f1, cx1, cy1)
    N = np.linalg.inv(K_of(f1, c1x, c1y)) @ Hm @ K_of(f0, W / 2, H / 2)
    M = N @ N.T
    s = np.sqrt(np.trace(M) / 3.0)
    R = N / s
    return f0, c1x, c1y, orth_resid(Hm, f0, f1, W / 2, H / 2, c1x, c1y), R, s


def main():
    files = sorted(f for f in os.listdir(TRUTH_DIR) if f.endswith(".lua"))
    print("H = s * K1 * R * K0^-1  orthogonality residual  (0 => model exact)\n")
    hdr = (f"{'file':<20} {'slot':>4} {'variant':<18} {'f0':>8} {'f1':>8} "
           f"{'resid':>10} {'rot_deg':>8}")
    print(hdr)
    print("-" * len(hdr))
    for fn in files:
        d = parse(os.path.join(TRUTH_DIR, fn))
        f35px = d["f35"] / SENSOR * max(W, H)
        for slot in (1, 2, 4):
            if slot not in d:
                continue
            Ht = to_pix(d[slot])
            Ht = Ht / Ht[2, 2]
            variants = [
                ("f1=f35 centre=stored", f35px, d["cx"] * W, d["cy"] * H, False, False),
                ("f1=f35 centre=image", f35px, W / 2, H / 2, False, False),
                ("f1=f35 centre=free", f35px, d["cx"] * W, d["cy"] * H, True, False),
                ("f1=f35 c=free f0=free", f35px, d["cx"] * W, d["cy"] * H, True, True),
            ]
            for label, f1, cx1, cy1, free_c, free_f0 in variants:
                f0, c1x, c1y, err, R, s = solve(Ht, f1, cx1, cy1, free_c, free_f0)
                ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                print(f"{fn:<20} {slot:4d} {label:<18} {f0:8.1f} {f1:8.1f} "
                      f"{err:10.2e} {ang:8.3f}", flush=True)


if __name__ == "__main__":
    main()
