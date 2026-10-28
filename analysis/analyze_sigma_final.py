"""
Final analysis of the dense pure-pitch sigma_v sweep (80 images, 4 seeds per
angle 2..40 deg).

Per image we measure, from the *known* camera and the Auto homography:
  init   mean signed tilt of the visible world verticals from image vertical
  resid  same after applying the Auto homography in pixel space
  conv   angle between the two extreme visible verticals (initial / after)

Correction fraction f = 1 - resid/init  (== corrected angle / true angle).
Fit f(theta) = lam/(lam+1), lam = L0 exp(-theta^2/(2 sigma^2)).

Three x-axes are fitted, because the solver's internal angle scale is not the
camera pitch for this renderer (the walls are close, so the visible keystone is
only ~0.15x the pitch):
  A  theta = camera pitch            -> sigma_pitch   (the requested number)
  B  theta = initial visible tilt    -> sigma_visible (solver-internal units)
  C  theta = initial convergence     -> sigma_conv    (comparable to the 15 deg literal)

Bootstrap CI resamples whole angles (cluster bootstrap), which is the honest
error bar given the image-to-image bimodality.
"""

import argparse
import json
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
RE_TR = re.compile(r'UprightTransform_(\d)\s*=\s*"([^"]*)"')
RE_F35 = re.compile(r'UprightFocalLength35mm\s*=\s*([-\d.eE]+)')
RE_UP = re.compile(r'PerspectiveUpright\s*=\s*([-\d.eE]+)')
VERTS = [(a, b) for a, b in world_segments()
         if abs(a[0] - b[0]) < 1e-9 and abs(a[2] - b[2]) < 1e-9]


def proj(a, rx):
    p = K @ (rot(-rx, 0, 0).T @ np.array(a, dtype=float))
    return p[:2] / p[2]


def measure(rx, Hp=None):
    before, after, dirs_b, dirs_a = [], [], [], []
    for a, b in VERTS:
        p1, p2 = proj(a, rx), proj(b, rx)
        ts = np.linspace(0, 1, 41)
        if sum(0 <= (1 - t) * p1[0] + t * p2[0] < W and
               0 <= (1 - t) * p1[1] + t * p2[1] < H for t in ts) < 8:
            continue
        d = p2 - p1
        before.append(np.degrees(np.arctan2(d[0], d[1])))
        if Hp is None:
            after.append(np.nan)
            continue
        q1, q2 = Hp @ np.array([*p1, 1]), Hp @ np.array([*p2, 1])
        if q1[2] <= 0.1 or q2[2] <= 0.1:
            after.append(np.nan)
            continue
        dw = q2[:2] / q2[2] - q1[:2] / q1[2]
        after.append(np.degrees(np.arctan2(dw[0], dw[1])))
    return np.array(before), np.array(after)


def conv_deg(dirs):
    d = np.array(dirs)
    if len(d) < 2:
        return np.nan
    return float(np.percentile(d, 95) - np.percentile(d, 5))


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
        tr = {int(n): np.array([float(v) for v in s.split(",")]).reshape(3, 3)
              for n, s in RE_TR.findall(text) if len(s.split(",")) == 9}
        if 1 not in tr:
            continue
        theta = float(base[1:3])
        rx = np.deg2rad(theta)
        init, auto = measure(rx, A.to_pix(tr[1], W, H))
        if len(init) < 3:
            continue
        sgn = np.sign(np.nanmean(init))
        keep = ~np.isnan(auto)
        if keep.sum() < 3:
            continue
        i0 = (init[keep] * sgn).mean()
        i1 = (auto[keep] * sgn).mean()
        full = np.nan
        if 2 in tr:
            _, fa = measure(rx, A.to_pix(tr[2], W, H))
            kf = ~np.isnan(fa)
            if kf.sum() >= 3:
                full = 1 - (fa[kf] * sgn).mean() / i0
        f35 = float(RE_F35.search(text).group(1))
        rows.append(dict(name=base, theta=theta, seed=int(base[-1]),
                         init=float(i0), resid=float(i1),
                         frac=float(1 - i1 / i0),
                         frac_full=float(full),
                         ratio=float((1 - i1 / i0) / full) if full and full > 1e-3 else np.nan,
                         conv0=conv_deg(init * sgn),
                         conv1=conv_deg(auto * sgn),
                         f35=f35, up=int(float(RE_UP.search(text).group(1)))))
    con.close()
    rows.sort(key=lambda r: (r["theta"], r["seed"]))
    return rows


def model(p, t):
    L0, s = np.exp(p[0]), np.exp(p[1])
    lam = L0 * np.exp(-t * t / (2 * s * s))
    return lam / (lam + 1.0)


def fit(t, f, p0=(3.0, 15.0)):
    def res(p):
        pred = np.clip(model(p, t), 1e-6, 1 - 1e-6)
        return (pred - f) / np.sqrt(pred * (1 - pred))
    r = least_squares(res, [np.log(p0[0]), np.log(p0[1])], method="trf", loss="soft_l1",
                      f_scale=1.0, bounds=([np.log(0.02), np.log(0.5)],
                                           [np.log(500.0), np.log(120.0)]), max_nfev=40000)
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def cluster_ci(t, f, groups, n=2000, seed=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        try:
            out.append(fit(t[idx], f[idx]))
        except Exception:
            pass
    out = np.array(out)
    return np.percentile(out[:, 0], [2.5, 97.5]), np.percentile(out[:, 1], [2.5, 97.5])


def report(tag, t, f, groups, out):
    ok = np.isfinite(t) & np.isfinite(f)
    t, f, groups = t[ok], f[ok], groups[ok]
    L0, s = fit(t, f)
    pred = model(np.log([L0, s]), t)
    maxres, rms = float(np.max(np.abs(f - pred))), float(np.sqrt(np.mean((f - pred) ** 2)))
    ciL, ciS = cluster_ci(t, f, groups)
    print(f"\n== {tag} ==  n={len(t)}")
    print(f"L0    = {L0:.3f}  CI {ciL[0]:.3f}..{ciL[1]:.3f}   asymptote {L0/(L0+1):.3f}")
    print(f"sigma = {s:.2f} deg  CI {ciS[0]:.2f}..{ciS[1]:.2f}   maxres {maxres:.3f}  rms {rms:.3f}")
    out[tag] = dict(L0=L0, sigma=s, ci_sigma=list(ciS), ci_L0=list(ciL),
                    maxres=maxres, rms=rms, n=int(len(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = load()
    good = [r for r in rows if r["up"] == 1 and np.isfinite(r["frac"])]
    print(f"images with Auto data: {len(rows)}  usable: {len(good)}")
    print(f"{'theta':>6}{'n':>3}{'init':>7}{'frac':>7}{'sd':>6}{'full':>7}{'ratio':>7}"
          f"{'conv0':>7}{'sat':>5}")
    ang = np.array(sorted({r["theta"] for r in good}))
    for a in ang:
        sub = [r for r in good if r["theta"] == a]
        fr = np.array([r["frac"] for r in sub])
        print(f"{a:6.0f}{len(sub):3d}{np.mean([r['init'] for r in sub]):7.2f}"
              f"{fr.mean():7.3f}{fr.std(ddof=1):6.3f}"
              f"{np.nanmean([r['frac_full'] for r in sub]):7.3f}"
              f"{np.nanmean([r['ratio'] for r in sub]):7.3f}"
              f"{np.mean([r['conv0'] for r in sub]):7.2f}"
              f"{sum(r['f35'] >= 39.9 for r in sub):5d}")

    t = np.array([r["theta"] for r in good])
    f = np.array([r["frac"] for r in good])
    groups = t.copy()
    out = {}
    report("A. fraction vs camera pitch", t, f, groups, out)
    report("B. fraction vs initial visible tilt", np.array([r["init"] for r in good]), f, groups, out)
    conv = np.array([r["conv0"] for r in good])
    report("C. fraction vs initial convergence", conv, f, groups, out)
    ratio = np.array([r["ratio"] for r in good])
    report("D. auto/full ratio vs pitch", t, ratio, groups, out)

    print("\nbranch split by focal-estimate saturation (f35 >= 39.9 means clamped):")
    for a in ang:
        sub = [r for r in good if r["theta"] == a]
        sat = [r["frac"] for r in sub if r["f35"] >= 39.9]
        fre = [r["frac"] for r in sub if r["f35"] < 39.9]
        print(f"  {a:4.0f}  n_sat={len(sat)}/{len(sub)}  "
              f"sat={np.mean(sat) if sat else float('nan'):.3f}  "
              f"free={np.mean(fre) if fre else float('nan'):.3f}")

    if args.json:
        json.dump(dict(rows=rows, fits=out), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
