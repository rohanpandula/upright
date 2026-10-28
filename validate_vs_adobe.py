"""
Ground-truth validation of the reverse-engineered Upright model against the
REAL Adobe Lightroom Classic solver.

Adobe's own solver was driven through the actual Lightroom Classic UI on four
synthetic images whose camera rotation is known exactly (see make_test_images.py).
Lightroom stored its results in the catalog (UprightTransform_0..5). This script
loads those stored transforms and compares them, on the same images, against our
reimplementation.

Metrics (all detector-free -- they use the KNOWN world geometry):

  residual_tilt_deg : angle of R_gt^T @ R_est where R_est is the rotation implied
                      by the homography (polar factor of K^-1 H K). 0 deg means
                      the camera tilt was perfectly removed; the un-corrected
                      image scores |R_gt| (e.g. 22.8 deg for case A).
  vert_conv_deg     : angle between the two extreme wall-A verticals after warping.
                      0 deg means the verticals are truly parallel.
  horiz_conv_deg    : same for the x-family horizontals.
  roll_resid_deg    : residual roll about the optical axis.
  p99_stretch       : 99th pct local area magnification (barrel/pincushion cost).
  crop_keep         : max inscribed axis-aligned area fraction.
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upright_core as uc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "lrtest", "in")
TRUTH_DIR = os.path.join(HERE, "adobe_truth")

W, H = 1400, 1000
F = 1600.0

# slot index == crs:PerspectiveUpright value
SLOTS = {"auto": 1, "full": 2, "level": 3, "vertical": 4}

CASES = {
    "A_clean": dict(img=6727297, text=6727297, rx=-0.384, ry=0.105, rz=0.0),
    "B_strong": dict(img=6727497, text=6727497, rx=-0.60, ry=0.21, rz=0.0),
    "C_noisy": dict(img=6727693, text=6727693, rx=-0.384, ry=0.105, rz=0.0),
    "D_roll": dict(img=6727887, text=6727887, rx=-0.384, ry=0.105, rz=0.052),
}


def rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def K_of(f, w, h):
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def parse_adobe(path):
    """Extract UprightTransform_N (normalized [0,1] coords) from the catalog blob."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r'UprightTransform_(\d)\s*=\s*"([^"]+)"', txt):
        vals = [float(v) for v in m.group(2).split(",")]
        if len(vals) == 9:
            out[int(m.group(1))] = np.array(vals).reshape(3, 3)
    return out


def to_pix(Hn, w, h):
    """normalized-[0,1] homography -> pixel homography."""
    S = np.diag([1.0 / w, 1.0 / h, 1.0])
    return np.linalg.inv(S) @ Hn @ S


def polar_rot(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def residual(R_gt, Hpix, f=F, w=W, h=H):
    """How much of the ground-truth camera tilt is left after applying Hpix."""
    K = K_of(f, w, h)
    M = np.linalg.inv(K) @ Hpix @ K
    R_est = polar_rot(M)
    D = R_gt.T @ R_est
    # decompose D to report each axis separately
    return rot_angle_deg(D)


def project(a, R_gt, f=F, w=W, h=H):
    K = K_of(f, w, h)
    p = K @ (R_gt.T @ np.asarray(a, dtype=float))
    return p[:2] / p[2]


def line_convergence(segA, segB, R_gt, Hpix):
    """Angle between two world lines after projection and warping."""
    def warped_dir(seg):
        p1 = project(seg[0], R_gt)
        p2 = project(seg[1], R_gt)
        q1 = Hpix @ np.array([p1[0], p1[1], 1.0])
        q2 = Hpix @ np.array([p2[0], p2[1], 1.0])
        a = q1[:2] / q1[2]
        b = q2[:2] / q2[2]
        return b - a
    d1 = warped_dir(segA)
    d2 = warped_dir(segB)
    n1 = d1 / (np.linalg.norm(d1) + 1e-12)
    n2 = d2 / (np.linalg.norm(d2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), -1, 1))))


WALL_A_V = [((-9, -5, 25), (-9, 20, 25)), ((9, -5, 25), (9, 20, 25))]
WALL_A_H = [((-9, -5, 25), (9, -5, 25)), ((-9, 20, 25), (9, 20, 25))]


def stretch_crop(Hpix, w=W, h=H):
    """Local area magnification p99 and max inscribed area fraction."""
    yy, xx = np.mgrid[0:h:8, 0:w:8]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], 0)
    Hi = np.linalg.inv(Hpix)
    q = Hi @ p
    qx, qy = q[0] / q[2], q[1] / q[2]
    # jacobian by finite differences on the inverse map
    q2 = Hi @ (p + np.array([[1], [0], [0]]))
    q3 = Hi @ (p + np.array([[0], [1], [0]]))
    j11 = q2[0] / q2[2] - qx
    j21 = q2[1] / q2[2] - qy
    j12 = q3[0] / q3[2] - qx
    j22 = q3[1] / q3[2] - qy
    det = np.abs(j11 * j22 - j12 * j21)
    stretch = float(np.percentile(det, 99))
    scale, _ = uc.constrain_crop(Hpix, w, h)
    return stretch, float(scale * scale)


def main():
    rows = []
    print(f"{'case':<9} {'mode':<9} {'who':<6} {'resid':>7} {'vert':>7} {'horiz':>7} "
          f"{'p99str':>7} {'crop%':>6}")
    print("-" * 68)
    for name, c in CASES.items():
        R_gt = rot(c["rx"], c["ry"], c["rz"])
        base = rot_angle_deg(R_gt)
        print(f"{name:<9} {'(none)':<9} {'orig':<6} {base:7.2f} "
              f"{line_convergence(*WALL_A_V, R_gt, np.eye(3)):7.2f} "
              f"{line_convergence(*WALL_A_H, R_gt, np.eye(3)):7.2f} "
              f"{1.0:7.3f} {100.0:6.1f}")
        truth_path = os.path.join(TRUTH_DIR, f"img_{c['text']}.lua")
        adobe = parse_adobe(truth_path)
        gray = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                          .open(os.path.join(IN_DIR, f"{name}.tif")).convert("L"),
                          dtype=np.float64)
        for mode, slot in SLOTS.items():
            for who, Hpix in (
                ("adobe", to_pix(adobe[slot], W, H) if slot in adobe else None),
            ):
                if Hpix is None:
                    continue
                Hpix = Hpix / Hpix[2, 2]
                st, ck = stretch_crop(Hpix)
                r = residual(R_gt, Hpix)
                print(f"{name:<9} {mode:<9} {who:<6} {r:7.2f} "
                      f"{line_convergence(*WALL_A_V, R_gt, Hpix):7.2f} "
                      f"{line_convergence(*WALL_A_H, R_gt, Hpix):7.2f} "
                      f"{st:7.3f} {100*ck:6.1f}")
                rows.append(dict(case=name, mode=mode, who=who, resid=r, stretch=st, crop=ck))
            out = uc.upright_transform(gray, f0=F, mode=mode)
            Ho = out["H"] / out["H"][2, 2]
            st, ck = stretch_crop(Ho)
            r = residual(R_gt, Ho)
            print(f"{name:<9} {mode:<9} {'ours':<6} {r:7.2f} "
                  f"{line_convergence(*WALL_A_V, R_gt, Ho):7.2f} "
                  f"{line_convergence(*WALL_A_H, R_gt, Ho):7.2f} "
                  f"{st:7.3f} {100*ck:6.1f}")
            rows.append(dict(case=name, mode=mode, who="ours", resid=r, stretch=st, crop=ck))
        print()
    with open(os.path.join(HERE, "adobe_validation.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote adobe_validation.json")


if __name__ == "__main__":
    main()
