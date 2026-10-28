"""
Measure sigma_v from the dense pure-pitch Auto sweep.

Ground truth: the camera is pitched by theta (pure pitch, ry = rz = 0), so the
*true* keystone angle is theta.  The corrected angle is obtained by projecting
the known world verticals through the true camera, applying the Auto homography
in pixel space, and measuring how much vertical tilt is removed:

    fraction(theta) = (tilt_initial - tilt_after) / tilt_initial
                    = corrected angle / true angle      (both in nominal units)

then fit

    fraction(theta) = lam/(lam+1),   lam = L0 * exp(-theta^2 / (2 sigma^2))

A cross-check metric uses the polar rotation angle of K^-1 H K with the focal
Lightroom itself reported.

Both are fitted by robust least squares with a bootstrap CI over images.
"""

import argparse
import json
import os
import re
import sqlite3

import numpy as np
from scipy.optimize import least_squares

import analyze_sigma_sweep as A
from make_test_images import rot, world_segments

DB = "/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat"
W, H, F = 1400, 1000, 1600.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
RE_NAME = re.compile(r'^V(\d\d)S(\d)$')
RE_TR = re.compile(r'UprightTransform_1\s*=\s*"([^"]*)"')
RE_F35 = re.compile(r'UprightFocalLength35mm\s*=\s*([-\d.eE]+)')
RE_UP = re.compile(r'PerspectiveUpright\s*=\s*([-\d.eE]+)')
VERTS = [(a, b) for a, b in world_segments()
         if abs(a[0] - b[0]) < 1e-9 and abs(a[2] - b[2]) < 1e-9]


def proj(a, rx):
    p = K @ (rot(-rx, 0, 0).T @ np.array(a, dtype=float))
    return p[:2] / p[2]


def tilts(rx, Hp=None):
    """Per-vertical tilt from image vertical, before and after Hp."""
    before, after = [], []
    for a, b in VERTS:
        p1, p2 = proj(a, rx), proj(b, rx)
        ts = np.linspace(0, 1, 41)
        if sum(0 <= (1 - t) * p1[0] + t * p2[0] < W and
               0 <= (1 - t) * p1[1] + t * p2[1] < H for t in ts) < 8:
            continue
        d = p2 - p1
        before.append(np.degrees(np.arctan2(abs(d[0]), abs(d[1]))))
        if Hp is None:
            continue
        q1, q2 = Hp @ np.array([*p1, 1]), Hp @ np.array([*p2, 1])
        if q1[2] <= 0.1 or q2[2] <= 0.1:
            continue
        dw = q2[:2] / q2[2] - q1[:2] / q1[2]
        after.append(np.degrees(np.arctan2(abs(dw[0]), abs(dw[1]))))
    return np.array(before), np.array(after)


def load():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = """
    SELECT f.baseName, d.text
    FROM Adobe_imageDevelopSettings d
    JOIN Adobe_images i ON i.id_local = d.image
    JOIN AgLibraryFile f ON f.id_local = i.rootFile
    JOIN AgLibraryFolder fo ON fo.id_local = f.folder
    JOIN AgLibraryRootFolder r ON r.id_local = fo.rootFolder
    WHERE d.text LIKE '%UprightTransform_1%'
      AND r.absolutePath || '/' || fo.pathFromRoot LIKE '%/lrtest/abstain%'
    """
    rows = []
    for base, text in con.execute(q):
        if not RE_NAME.match(base):
            continue
        tm, upm = RE_TR.search(text), RE_UP.search(text)
        if not tm or not upm:
            continue
        Hn = np.array([float(v) for v in tm.group(1).split(",")]).reshape(3, 3)
        f35 = float(RE_F35.search(text).group(1))
        theta = float(base[1:3])
        before, after = tilts(np.deg2rad(theta), A.to_pix(Hn, W, H))
        if len(before) < 3 or len(after) < 3:
            continue
        frac_phys = 1.0 - after.mean() / before.mean()
        rmag = A.decomp(Hn, f35, 0.5 * W, 0.5 * H)[0]
        rows.append(dict(name=base, theta=theta, seed=int(base[-1]),
                         init=float(before.mean()), resid=float(after.mean()),
                         frac_phys=float(frac_phys), frac_rot=rmag / theta,
                         rot=rmag, f35=f35, mag=float(np.max(np.abs(Hn - np.eye(3)))),
                         up=int(float(upm.group(1)))))
    con.close()
    rows.sort(key=lambda r: (r["theta"], r["seed"]))
    return rows


def model(p, t):
    L0, s = np.exp(p[0]), np.exp(p[1])
    lam = L0 * np.exp(-t * t / (2 * s * s))
    return lam / (lam + 1.0)


def fit(t, f):
    def res(p):
        pred = np.clip(model(p, t), 1e-6, 1 - 1e-6)
        return (pred - f) / np.sqrt(pred * (1 - pred))
    r = least_squares(res, [np.log(3.0), np.log(15.0)], method="trf", loss="soft_l1",
                      f_scale=1.0, bounds=([np.log(0.02), np.log(1.0)],
                                           [np.log(200.0), np.log(80.0)]), max_nfev=40000)
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def report(tag, t, f, out):
    L0, s = fit(t, f)
    pred = model(np.log([L0, s]), t)
    maxres = float(np.max(np.abs(f - pred)))
    rms = float(np.sqrt(np.mean((f - pred) ** 2)))
    rng = np.random.default_rng(11)
    bs = []
    for _ in range(3000):
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
    print(f"\n== {tag} ==")
    print(f"L0    = {L0:.4f}   95% CI {lo0:.4f}..{hi0:.4f}   asymptote {L0/(L0+1):.3f}")
    print(f"sigma = {s:.3f} deg 95% CI {lo:.3f}..{hi:.3f} deg   n={len(t)}")
    print(f"max residual {maxres:.4f}  rms {rms:.4f}   fitted/15deg = {s/15.0:.3f}")
    out[tag] = dict(L0=L0, sigma=s, ci=[lo, hi], ci_L0=[lo0, hi0],
                    maxres=maxres, rms=rms, n=len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = load()
    print(f"usable Auto rows: {len(rows)}  (PerspectiveUpright==1: "
          f"{sum(r['up'] == 1 for r in rows)})")
    print(f"{'name':<8}{'theta':>6}{'f35':>7}{'init':>7}{'resid':>7}"
          f"{'frac_phys':>10}{'frac_rot':>9}{'mag':>10}")
    for r in rows:
        print(f"{r['name']:<8}{r['theta']:6.1f}{r['f35']:7.2f}{r['init']:7.2f}"
              f"{r['resid']:7.2f}{r['frac_phys']:10.3f}{r['frac_rot']:9.3f}{r['mag']:10.3e}")

    t = np.array([r["theta"] for r in rows])
    fp = np.clip(np.array([r["frac_phys"] for r in rows]), 0, 1)
    fr = np.array([r["frac_rot"] for r in rows])
    out = {}
    report("physical keystone removal", t, fp, out)
    report("H rotation angle", t, fr, out)

    # per-angle medians as a robustness check
    ang = np.unique(t)
    med = np.array([np.median(fp[t == a]) for a in ang])
    report("physical, per-angle medians", ang, med, out)

    # branch split: focal estimate clamped at 40 vs free
    print("\nabstention-like branch (f35 >= 39.9, i.e. focal estimate saturated):")
    for a in ang:
        sub = [r for r in rows if r["theta"] == a]
        print(f"  {a:4.0f} deg  n_sat={sum(r['f35'] >= 39.9 for r in sub)}/{len(sub)}  "
              f"frac_sat={np.mean([r['frac_phys'] for r in sub if r['f35'] >= 39.9]) if any(r['f35'] >= 39.9 for r in sub) else float('nan'):.3f}  "
              f"frac_free={np.mean([r['frac_phys'] for r in sub if r['f35'] < 39.9]) if any(r['f35'] < 39.9 for r in sub) else float('nan'):.3f}")

    lit = np.deg2rad(15.0)
    s = out["physical keystone removal"]["sigma"]
    ci = out["physical keystone removal"]["ci"]
    print(f"\nbinary literal 0.2618 rad = 15.000 deg; fitted sigma/15deg = {s/15.0:.3f}; "
          f"CI excludes 15.00: {ci[1] < 15.0 or ci[0] > 15.0}")
    print(f"literal sigma = {lit:.6f} rad = 15.000 deg")
    if args.json:
        json.dump(dict(rows=rows, fits=out), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
