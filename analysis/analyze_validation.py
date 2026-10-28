"""
Analyse the Lightroom probe output and validate the reference implementation.

Reads lrtest/result.json (written by the Lightroom plugin), pulls out the
crs:UprightTransform_N homographies Adobe's own solver produced, converts them
from normalised frame coordinates to pixels, and scores them.

Metrics per (image, mode):
  vert_tilt   mean residual tilt of detected vertical-family lines after the
              correction (deg). Lower = more fully rectified.
  horiz_tilt  same for horizontal-family lines.
  in_frac     fraction of warped pixels still inside the source frame.
  vs_ours     mean displacement between Adobe's mapping and ours, normalised
              by the image diagonal. 0 = identical.

A pure-rotation check tests the [RE] claim that Adobe's transform is K R K^-1
rather than a general projective fit.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from upright_core import (  # noqa: E402
    detect_lines, estimate_vanishing_points, classify_families,
    upright_transform, intrinsics, homography_from_params, _normalize,
)

LT = os.path.join(HERE, "lrtest")
W, H_IMG = 1400, 1000
F = 1600.0
MODES = ["off", "auto", "level", "vertical", "full"]
IDX = {"off": 0, "auto": 1, "full": 2, "level": 3, "vertical": 4, "guided": 5}


def norm_to_pixel(Hn, w=W, h=H_IMG):
    D = np.diag([float(w), float(h), 1.0])
    return D @ Hn @ np.linalg.inv(D)


def parse_transform(s):
    """Adobe stores nine comma/space separated numbers, row-major."""
    if s is None:
        return None
    nums = [float(t) for t in str(s).replace(",", " ").split()]
    if len(nums) != 9:
        return None
    return np.array(nums).reshape(3, 3)


def line_tilt(H, segs, vertical=True):
    if len(segs) == 0:
        return float("nan")
    l = np.array([np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0]) for s in segs])
    lt = (np.linalg.inv(H).T @ l.T).T
    d = np.stack([-lt[:, 1], lt[:, 0]], axis=1)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    a = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1]))) if vertical \
        else np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    return float(np.mean(a))


def inside_fraction(H, w=W, h=H_IMG, n=40):
    yy, xx = np.mgrid[0:h:n, 0:w:n]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    q = (H @ p.T).T
    Wc = np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])
    sx, sy = q[:, 0] / Wc, q[:, 1] / Wc
    return float(np.mean((sx >= 0) & (sx <= w) & (sy >= 0) & (sy <= h)))


def displacement(Ha, Hb, w=W, h=H_IMG, n=30):
    """Mean |Hb^-1(Ha(p)) - p| / diagonal -- how differently the two map points."""
    yy, xx = np.mgrid[0:h:n, 0:w:n]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    qa = (Ha @ p.T).T
    qb = (np.linalg.inv(Hb) @ qa.T).T
    Wc = np.where(np.abs(qb[:, 2]) < 1e-12, 1e-12, qb[:, 2])
    r = qb[:, :2] / Wc
    d = np.linalg.norm(r - p[:, :2], axis=1)
    return float(np.mean(d) / np.hypot(w, h))


def is_pure_rotation(H, w=W, h=H_IMG):
    """Fit K1 R K0^-1 (3 rot + 1 focal) and return the relative fit error."""
    from scipy.optimize import least_squares
    K0 = intrinsics(F, F, w / 2.0, h / 2.0)
    yy, xx = np.mgrid[0:h:25, 0:w:25]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    q = (H @ p.T).T
    q = q[:, :2] / np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])[:, None]

    def resid(x):
        Hf = homography_from_params(np.array([x[0], x[0], x[1], x[2], x[3], 0, 0]), K0)
        r = (Hf @ p.T).T
        r = r[:, :2] / np.where(np.abs(r[:, 2]) < 1e-12, 1e-12, r[:, 2])[:, None]
        return (r - q).ravel() / np.hypot(w, h)

    x0 = np.array([F, 0.0, 0.0, 0.0])
    try:
        s = least_squares(resid, x0, max_nfev=400)
        e = float(np.sqrt(np.mean(resid(s.x) ** 2)))
        return e, s.x[0] / F
    except Exception as exc:  # noqa: BLE001
        return float("nan"), float("nan")


def main():
    rp = os.path.join(LT, "result.json")
    if not os.path.exists(rp):
        print("no result.json yet")
        return
    data = json.load(open(rp))
    print(f"catalog: {data.get('catalog')}")
    print(f"images: {len(data.get('images', []))}\n")

    for entry in data.get("images", []):
        path = os.path.join(LT, "in", entry["image"])
        gray = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(path).convert("L"))
        segs = detect_lines(gray)
        vps = estimate_vanishing_points(segs, W, H_IMG)
        fam = classify_families(vps, segs)
        vert = fam["vertical"]
        horiz = fam["horizontals"]
        vsegs = segs[vert["inliers"]] if vert else np.zeros((0, 4))
        hsegs = segs[horiz[0]["inliers"]] if horiz else np.zeros((0, 4))

        print(f"=== {entry['image']}  ({len(segs)} segments, "
              f"{len(vsegs)}V/{len(hsegs)}H inliers) ===")
        print(f"  {'source':<22} {'v_tilt':>8} {'h_tilt':>8} {'in%':>6} {'pure_rot_err':>13} {'f_ratio':>8}")

        def row(label, Hm):
            print(f"  {label:<22} {line_tilt(Hm, vsegs, True):8.3f} "
                  f"{line_tilt(Hm, hsegs, False):8.3f} "
                  f"{inside_fraction(Hm)*100:6.1f} "
                  f"{is_pure_rotation(Hm)[0]:13.5f} {is_pure_rotation(Hm)[1]:8.4f}")

        row("input (identity)", np.eye(3))

        mine = {}
        for mode in ("auto", "level", "vertical", "full"):
            try:
                r = upright_transform(gray.astype(np.uint8)
                                      if gray.dtype != np.uint8 else gray, f0=F, mode=mode)
                mine[mode] = r["H"]
                row(f"ours:{mode}", r["H"])
            except Exception as exc:  # noqa: BLE001
                print(f"  ours:{mode} failed: {exc}")

        for rec in entry["modes"]:
            mode = rec["mode"]
            s = rec.get("settings", {})
            key = f"UprightTransform_{IDX.get(mode, -1)}"
            Hn = parse_transform(s.get(key))
            extra = (f"  PU={s.get('PerspectiveUpright')} "
                     f"V={s.get('PerspectiveVertical')} "
                     f"H={s.get('PerspectiveHorizontal')} "
                     f"R={s.get('PerspectiveRotate')}")
            if Hn is None:
                keys = [k for k in s if k.startswith("Upright")]
                print(f"  Adobe:{mode:<16} NO TRANSFORM  {extra}")
                if keys:
                    print(f"      upright keys present: {keys}")
                continue
            Hp = norm_to_pixel(Hn)
            row(f"Adobe:{mode}", Hp)
            print(f"      {extra}")
            if mode in mine:
                print(f"      displacement vs ours:{mode} = "
                      f"{displacement(Hp, mine[mode])*100:.3f}% of diagonal")
        print()


if __name__ == "__main__":
    main()
