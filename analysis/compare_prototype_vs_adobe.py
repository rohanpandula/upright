"""
Head-to-head: run the clean-room prototype on the *same* photographs Lightroom
already solved, and compare its output with Adobe's cached Upright transforms.

Ground truth = UprightTransform_N written by the shipped binary.
Subject      = research/upright/upright_core.py

Metrics per (photo, mode):
  disp%   mean point displacement between Adobe's mapping and ours, as a
          percentage of the image diagonal. Lower = more alike.
  dv/dh   difference in recovered pitch / yaw (degrees).
  tilt_ad tilt of detected vertical lines after Adobe's transform (deg).
  tilt_us same after ours.  Both should be small if both rectify.

Caveat: Adobe's transform was computed on the post-lens-correction render and
against its own line detector, so agreement is bounded by those differences --
they are not the same detector.
"""

import json
import os
import random
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from upright_core import detect_lines, upright_transform, classify_families, \
    estimate_vanishing_points
from test_rotation_model import fit_rotation, to_pixel, IDENT

MODES = {"auto": 1, "full": 2, "level": 3, "vertical": 4}
MAXSIDE = 1200


def _unit_area(H, w, h):
    """Rescale H so it maps the image to unit area, removing crop/zoom.

    Without this, disp_percent is dominated by how much each transform scales
    the frame: two transforms with identical shape correction but different
    crop score as badly as two genuinely different corrections.
    """
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    q = (H @ corners.T).T
    q = q[:, :2] / q[:, 2:3]
    area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
    k = 1.0 / np.sqrt(max(area, 1e-12))
    S = np.diag([k, k, 1.0])
    return S @ H


def disp_percent(Ha, Hb, w, h, n=24):
    """Scale-invariant mean point displacement between two mappings, as % of diagonal.

    Both transforms are area-normalised first, so the result reflects SHAPE
    disagreement only (residual convergence, shear, perspective) and not how
    much either one crops.
    """
    Ha = _unit_area(Ha, w, h)
    Hb = _unit_area(Hb, w, h)
    yy, xx = np.mgrid[0:h:n, 0:w:n]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1).astype(float)
    qa = (Ha @ p.T).T
    qb = (np.linalg.inv(Hb) @ qa.T).T
    z = np.where(np.abs(qb[:, 2]) < 1e-12, 1e-12, qb[:, 2])
    r = qb[:, :2] / z[:, None]
    # centre both frames so translation does not masquerade as shape error
    r = r - r.mean(axis=0)
    p2 = p[:, :2] - p[:, :2].mean(axis=0)
    return float(np.mean(np.linalg.norm(r - p2, axis=1)) / np.hypot(w, h) * 100)


def tilt(H, segs, vertical=True):
    if len(segs) == 0:
        return float("nan")
    l = np.array([np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0]) for s in segs])
    lt = (np.linalg.inv(H).T @ l.T).T
    d = np.stack([-lt[:, 1], lt[:, 0]], axis=1)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    a = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1]))) if vertical \
        else np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    return float(np.mean(a))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lr_all.jsonl"
    n_target = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    recs = [json.loads(l) for l in open(src)]
    random.seed(7)
    random.shuffle(recs)

    rows = []
    for rec in recs:
        if len(rows) >= n_target:
            break
        if not os.path.exists(rec["path"]):
            continue
        v = rec["transform"].get("1")
        Hn = np.array(v, dtype=float).reshape(3, 3)
        if np.max(np.abs(Hn - IDENT)) < 1e-9:
            continue                                   # solver abstained
        try:
            im = Image.open(rec["path"])
            im.draft("RGB", (MAXSIDE, MAXSIDE))
            gray = np.asarray(im.convert("L"))
        except Exception:
            continue
        if gray.shape[0] < 300 or gray.shape[1] < 300:
            continue
        h, w = gray.shape
        segs = detect_lines(gray)
        if len(segs) < 20:
            continue
        vps = estimate_vanishing_points(segs, w, h)
        fam = classify_families(vps, segs)
        vs = fam["vertical"]
        vsegs = segs[vs["inliers"]] if vs else np.zeros((0, 4))
        if len(vsegs) < 6:
            continue

        out = {"path": os.path.basename(rec["path"]), "modes": {}}
        for mode, slot in MODES.items():
            sv = rec["transform"].get(str(slot))
            if sv is None:
                continue
            Ha = to_pixel(np.array(sv, dtype=float).reshape(3, 3), w, h)
            try:
                res = upright_transform(gray, f0=w, mode=mode)
                Hu = res["H"]
            except Exception as exc:
                out["modes"][mode] = {"err": str(exc)}
                continue
            try:
                _, ang_a, _ = fit_rotation(Ha, w, h)
                _, ang_u, _ = fit_rotation(Hu, w, h)
                dv = float(abs(ang_a[0] - ang_u[0]))
                dh = float(abs(ang_a[1] - ang_u[1]))
            except Exception:
                dv = dh = float("nan")
            out["modes"][mode] = {
                "disp": disp_percent(Ha, Hu, w, h),
                "dv": dv, "dh": dh,
                "tilt_adobe": tilt(Ha, vsegs),
                "tilt_ours": tilt(Hu, vsegs),
            }
        rows.append(out)
        print(f"  done {len(rows):3d}  {os.path.basename(rec['path'])[:40]}")

    print(f"\n{'mode':<10} {'n':>4} {'disp% med':>10} {'disp% p90':>10} "
          f"{'tilt_adobe':>11} {'tilt_ours':>10} {'dAngle med':>11}")
    for mode in MODES:
        vals = [r["modes"][mode] for r in rows if mode in r["modes"] and "disp" in r["modes"][mode]]
        if not vals:
            continue
        d = np.array([v["disp"] for v in vals])
        ta = np.array([v["tilt_adobe"] for v in vals])
        tu = np.array([v["tilt_ours"] for v in vals])
        da = np.array([np.hypot(v["dv"], v["dh"]) for v in vals])
        print(f"{mode:<10} {len(vals):4d} {np.median(d):10.3f} {np.percentile(d,90):10.3f} "
              f"{np.median(ta):11.3f} {np.median(tu):10.3f} {np.median(da):11.3f}")

    with open(os.path.join(HERE, "lr_comparison.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote lr_comparison.json ({len(rows)} photos)")


if __name__ == "__main__":
    main()
