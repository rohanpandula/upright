"""
Measure sigma_v from the dense pure-pitch Auto sweep.

For each image:
  true tilt      theta  (the pitch the scene was rendered at)
  corrected tilt theta_c = rotation angle of polar(K^-1 H K), H = UprightTransform_1

correction fraction f = theta_c / theta, then fit

    f(theta) = lam/(lam+1),   lam = L0 * exp(-theta^2 / (2 sigma^2))

by robust least squares, with a bootstrap CI over images.
"""

import argparse
import os
import re
import sqlite3

import numpy as np
from scipy.optimize import least_squares

DB = "/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat"
W, H = 1400, 1000
SENSOR = 36.0
RE_TR = re.compile(r'UprightTransform_1\s*=\s*"([^"]*)"')
RE_F35 = re.compile(r'UprightFocalLength35mm\s*=\s*([-\d.eE]+)')
RE_CX = re.compile(r'UprightCenterNormX\s*=\s*([-\d.eE]+)')
RE_CY = re.compile(r'UprightCenterNormY\s*=\s*([-\d.eE]+)')
RE_UP = re.compile(r'PerspectiveUpright\s*=\s*([-\d.eE]+)')
RE_NAME = re.compile(r'^V(\d\d)S(\d)$')


def to_pix(Hn, w, h):
    S = np.diag([1.0 / w, 1.0 / h, 1.0])
    return np.linalg.inv(S) @ Hn @ S


def polar_rot(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def decomp(Hn, f35, cx, cy):
    f = f35 / SENSOR * max(W, H)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R = polar_rot(np.linalg.inv(K) @ to_pix(Hn, W, H) @ K)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    rx = np.degrees(np.arctan2(-R[1, 2], R[2, 2]))
    ry = np.degrees(np.arcsin(np.clip(R[0, 2], -1, 1)))
    rz = np.degrees(np.arctan2(-R[0, 1], R[0, 0]))
    return ang, rx, ry, rz


def load(prefix="V"):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = """
    SELECT d.image, d.text, f.baseName, i.fileWidth, i.fileHeight
    FROM Adobe_imageDevelopSettings d
    JOIN Adobe_images i ON i.id_local = d.image
    JOIN AgLibraryFile f ON f.id_local = i.rootFile
    JOIN AgLibraryFolder fo ON fo.id_local = f.folder
    JOIN AgLibraryRootFolder r ON r.id_local = fo.rootFolder
    WHERE d.text LIKE '%UprightTransform_1%'
      AND r.absolutePath || '/' || fo.pathFromRoot LIKE '%/lrtest/abstain%'
    """
    rows = []
    for image, text, base, fw, fh in con.execute(q):
        m = RE_NAME.match(base)
        if not m:
            continue
        tm = RE_TR.search(text)
        if not tm:
            continue
        vals = [float(v) for v in tm.group(1).split(",")]
        if len(vals) != 9:
            continue
        Hn = np.array(vals).reshape(3, 3)
        f35m, cxm, cym = RE_F35.search(text), RE_CX.search(text), RE_CY.search(text)
        f35 = float(f35m.group(1)) if f35m else 35.0
        cx = (float(cxm.group(1)) if cxm else 0.5) * fw
        cy = (float(cym.group(1)) if cym else 0.5) * fh
        up = RE_UP.search(text)
        ang, rx, ry, rz = decomp(Hn, f35, cx, cy)
        rows.append(dict(image=image, name=base, theta=float(m.group(1)),
                         seed=int(m.group(2)), mag=float(np.max(np.abs(Hn - np.eye(3)))),
                         frac=ang / float(m.group(1)), ang=ang, rx=rx, ry=ry, rz=rz,
                         f35=f35, up=int(float(up.group(1))) if up else -1))
    con.close()
    rows.sort(key=lambda r: (r["theta"], r["seed"]))
    return rows


def model(p, t):
    L0, s = np.exp(p[0]), np.exp(p[1])
    lam = L0 * np.exp(-t * t / (2 * s * s))
    return lam / (lam + 1.0)


def fit(t, f, p0=(np.log(1.7), np.log(8.0))):
    def res(p):
        return (model(p, t) - f) / np.maximum(0.02, np.sqrt(model(p, t) * (1 - model(p, t)) / 1.0))
    r = least_squares(res, p0, method="trf", loss="soft_l1", f_scale=1.0,
                      bounds=([np.log(0.05), np.log(1.0)], [np.log(50.0), np.log(60.0)]),
                      max_nfev=20000)
    return np.exp(r.x[0]), np.exp(r.x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = load()
    good = [r for r in rows if r["up"] == 1]
    print(f"rows with Auto transform: {len(rows)}   PerspectiveUpright==1: {len(good)}")
    print(f"{'name':<9} {'theta':>6} {'mag':>9} {'corr':>7} {'frac':>6} "
          f"{'rx':>7} {'ry':>6} {'rz':>6} {'f35':>7}")
    for r in good:
        print(f"{r['name']:<9} {r['theta']:6.1f} {r['mag']:9.3e} {r['ang']:7.3f} "
              f"{r['frac']:6.3f} {r['rx']:7.3f} {r['ry']:6.3f} {r['rz']:6.3f} {r['f35']:7.2f}")

    t = np.array([r["theta"] for r in good])
    f = np.array([r["frac"] for r in good])
    L0, s = fit(t, f)
    pred = model(np.log([L0, s]), t)
    maxres = float(np.max(np.abs(f - pred)))
    rms = float(np.sqrt(np.mean((f - pred) ** 2)))

    rng = np.random.default_rng(7)
    bs = []
    for _ in range(4000):
        idx = rng.integers(0, len(t), len(t))
        if len(np.unique(t[idx])) < 6:
            continue
        try:
            bs.append(fit(t[idx], f[idx]))
        except Exception:
            pass
    bs = np.array(bs)
    lo, hi = np.percentile(bs[:, 1], [2.5, 97.5])
    lo0, hi0 = np.percentile(bs[:, 0], [2.5, 97.5])

    print(f"\nL0    = {L0:.4f}   (95% CI {lo0:.4f} .. {hi0:.4f})")
    print(f"sigma = {s:.3f} deg (95% CI {lo:.3f} .. {hi:.3f} deg)")
    print(f"max residual = {maxres:.4f}   rms = {rms:.4f}   n = {len(t)}")
    print(f"binary literal 0.2618 rad = 15.000 deg; fitted/literal = {s/15.0:.3f}; "
          f"CI excludes 15.00: {hi < 15.0 or lo > 15.0}")
    print(f"asymptote L0/(L0+1) = {L0/(L0+1):.3f}")
    if args.json:
        import json
        json.dump(dict(rows=good, L0=L0, sigma=s, ci=[lo, hi], maxres=maxres), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
