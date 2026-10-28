"""
Generate controlled test images for validating Lightroom's Upright against the
reference implementation.

A 3-family Manhattan scene (two walls meeting at a vertical corner) is projected
through a *known* camera, so the ground-truth rotation is known exactly.
Rendered as sRGB RGB TIFFs so Lightroom treats them as ordinary photographs.

Cases:
  A_clean    pitch 22 deg, yaw 6  - easy, clean
  B_strong   pitch 34 deg, yaw 12 - strong tilt, the "does Auto over-correct?" case
  C_noisy    A + gaussian noise + mild JPEG-ish softening - robustness
  D_roll     A + 3 deg roll - tests Level specifically

NON-MANHATTAN EXTENSION (2026-10-27)
Every earlier scene was two-wall Manhattan. These four families are the
out-of-distribution probe, all at pitch 15 deg / yaw 6 deg so the families are
comparable and only the *structure* varies:

  (a) NM_single_wall   one dominant plane, open on the other side. Verticals
                       only; the wall's horizontals are parallel to the image
                       plane, so the horizontal vanishing point is at infinity
                       and yaw is not observable from it.
  (b) NM_railing       the same wall plus railings running into depth, which
                       restores the second vanishing point -> yaw is observable.
  (c) NM_curves        wall plus two big circular arcs on it and a cylindrical
                       column. Curves contribute no straight edges; the arcs are
                       returned as ground-truth 3-D sample points so their
                       straightness can be measured before/after warping.
  (d) NM_three_plane   three mutually perpendicular planes (back wall, side
                       wall, floor) - the closest thing to a Manhattan corner.

`render_scene` returns (image, arcs) so callers can measure the known arcs.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

W, H = 1400, 1000
F = 1600.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lrtest", "in")


def rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def world_segments():
    """Two perpendicular walls -> vertical family, x family, z family."""
    segs = []
    for x in np.arange(-9, 9.01, 1.5):                       # verticals on wall A
        segs.append(((x, -5, 25), (x, 20, 25)))
    for y in np.arange(-5, 20.01, 1.25):                     # x-family horizontals
        segs.append(((-9, y, 25), (9, y, 25)))
    for z in np.arange(25, 80.01, 1.5):                      # verticals on wall B
        segs.append(((9, -5, z), (9, 20, z)))
    for y in np.arange(-5, 20.01, 1.25):                     # z-family horizontals
        segs.append(((9, y, 25), (9, y, 80)))
    return segs


def render(rx, ry, rz, w=W, h=H, f=F):
    img = Image.new("RGB", (w, h), (238, 236, 232))
    d = ImageDraw.Draw(img)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
    R = rot(rx, ry, rz)
    for a, b in world_segments():
        pa = K @ (R.T @ np.array(a, dtype=float))
        pb = K @ (R.T @ np.array(b, dtype=float))
        if pa[2] <= 0.2 or pb[2] <= 0.2:
            continue
        p1 = tuple(pa[:2] / pa[2])
        p2 = tuple(pb[:2] / pb[2])
        # alternate wall shading so the two planes read differently
        col = (46, 44, 42) if abs(a[0] - 9) < 1e-6 else (58, 56, 54)
        d.line([p1, p2], fill=col, width=4)
    return np.asarray(img, dtype=np.float64)


def add_grain(a, sigma=7.0, seed=3):
    rng = np.random.default_rng(seed)
    return np.clip(a + rng.normal(0, sigma, a.shape), 0, 255)


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = {
        "A_clean": (-0.384, 0.105, 0.0),      # pitch 22, yaw 6
        "B_strong": (-0.60, 0.21, 0.0),       # pitch 34, yaw 12
        "C_noisy": (-0.384, 0.105, 0.0),
        "D_roll": (-0.384, 0.105, 0.052),     # +3 deg roll
    }
    for name, (rx, ry, rz) in cases.items():
        a = render(rx, ry, rz)
        if name == "C_noisy":
            a = add_grain(a)
        Image.fromarray(a.astype(np.uint8)).save(
            os.path.join(OUT, f"{name}.tif"), compression="tiff_lzw")
        print(f"wrote {name}.tif  rx={rx} ry={ry} rz={rz}")
    print("ground truth (rad):", cases)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# non-Manhattan families
# --------------------------------------------------------------------------

PITCH15 = -0.2617993877991494      # 15 deg up
YAW6 = 0.10471975511965977         # 6 deg
NM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lrtest", "nm")


def _grid(segs, plane, fixed, u_range, v_range, du, dv, col):
    """Emit a grid on an axis-aligned plane. `plane` is the normal axis name.
    Family tags: 'v' vertical, 'h' horizontal (perpendicular to the optical
    axis, so a rectification makes it horizontal), 'd' depth (along the optical
    axis, so a rectification makes it converge on the principal point)."""
    def pt(u, v):
        return {"x": (fixed, u, v), "y": (u, fixed, v), "z": (u, v, fixed)}[plane]
    fam_u, fam_v = {"x": ("d", "v"), "y": ("h", "d"), "z": ("h", "v")}[plane]
    for u in np.arange(u_range[0], u_range[1] + 1e-9, du):
        segs.append((pt(u, v_range[0]), pt(u, v_range[1]), col, fam_v))
    for v in np.arange(v_range[0], v_range[1] + 1e-9, dv):
        segs.append((pt(u_range[0], v), pt(u_range[1], v), col, fam_u))


def _arc(center, r, plane_fixed, a0, a1, n, col, segs, arcs, label):
    cx, cy = center
    pts = []
    for t in np.linspace(a0, a1, n):
        pts.append((cx + r * np.cos(t), cy + r * np.sin(t), plane_fixed))
    arcs[label] = np.array(pts, dtype=float)
    for i in range(len(pts) - 1):
        segs.append((pts[i], pts[i + 1], col, "c"))


WALL = (58, 56, 54)
WALL2 = (44, 42, 41)
RAIL = (30, 29, 28)
ARC = (24, 23, 22)
FLOOR = (74, 71, 68)


def scene_single_wall():
    """(a) one dominant plane, nothing on the other side."""
    segs = []
    _grid(segs, "z", 30.0, (-14, 18), (-8, 20), 1.5, 1.25, WALL)
    return segs, {}


def scene_railing():
    """(b) the same wall, plus railings running into depth (second VP)."""
    segs = []
    _grid(segs, "z", 32.0, (-14, 18), (-8, 20), 1.5, 1.25, WALL)
    for x in (-6.0, 6.0):
        for y in (-0.5, 0.8, 2.1):
            segs.append(((x, y, 7.0), (x, y, 24.0), RAIL, "d"))
        for z in np.arange(7.0, 24.01, 2.4):
            segs.append(((x, -0.5, z), (x, 2.1, z), RAIL, "v"))
    return segs, {}


def scene_curves():
    """(c) two big arcs on the wall and a cylindrical column - the arcs and the
    rings are the only curved edges, everything else stays a straight line."""
    segs, arcs = [], {}
    _grid(segs, "z", 32.0, (-14, 18), (-8, 20), 3.0, 2.5, WALL)
    _arc((2.0, 8.0), 7.0, 32.0, 0.0, np.pi, 90, ARC, segs, arcs, "wall_arc_r7")
    _arc((2.0, 8.0), 5.0, 32.0, 0.0, np.pi, 90, ARC, segs, arcs, "wall_arc_r5")
    # cylindrical column: verticals around the barrel + curved rings
    cx, cz, r = -4.0, 18.0, 1.6
    for t in np.arange(0.0, 2 * np.pi, np.pi / 8):
        x, z = cx + r * np.cos(t), cz + r * np.sin(t)
        segs.append(((x, -2.0, z), (x, 16.0, z), WALL2, "v"))
    for y in (4.0, 7.0, 10.0):
        pts = [(cx + r * np.cos(t), y, cz + r * np.sin(t))
               for t in np.linspace(0, 2 * np.pi, 33)]
        arcs[f"column_ring_{int(y)}"] = np.array(pts, dtype=float)
        for i in range(len(pts) - 1):
            segs.append((pts[i], pts[i + 1], WALL2, "c"))
    return segs, arcs


def scene_three_plane():
    """(d) three mutually perpendicular planes."""
    segs = []
    _grid(segs, "z", 34.0, (-12, 20), (-8, 20), 1.5, 1.5, WALL)
    _grid(segs, "x", -12.0, (14, 34), (-8, 20), 1.5, 1.5, WALL2)
    _grid(segs, "y", -0.6, (-12, 20), (14, 34), 1.5, 1.5, FLOOR)
    return segs, {}


def scene_manhattan():
    """Control: four mutually perpendicular planes at the same 15/6 pose, i.e.
    the Manhattan corner the earlier experiments always used."""
    segs = []
    _grid(segs, "z", 34.0, (-12, 12), (-8, 20), 1.5, 1.5, WALL)
    _grid(segs, "x", -12.0, (14, 34), (-8, 20), 1.5, 1.5, WALL2)
    _grid(segs, "x", 12.0, (14, 34), (-8, 20), 1.5, 1.5, RAIL)
    _grid(segs, "y", -0.6, (-12, 12), (14, 34), 1.5, 1.5, FLOOR)
    return segs, {}


SCENES = {
    "NM_single_wall": scene_single_wall,
    "NM_railing": scene_railing,
    "NM_curves": scene_curves,
    "NM_three_plane": scene_three_plane,
    "CT_manhattan2": scene_manhattan,
}


def render_scene(name, rx=PITCH15, ry=YAW6, rz=0.0, w=W, h=H, f=F):
    """Render one non-Manhattan family. Returns (uint8 image, arcs dict)."""
    segs, arcs = SCENES[name]()
    img = Image.new("RGB", (w, h), (238, 236, 232))
    d = ImageDraw.Draw(img)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
    R = rot(rx, ry, rz)
    for a, b, col, fam in segs:
        pa = K @ (R.T @ np.array(a, dtype=float))
        pb = K @ (R.T @ np.array(b, dtype=float))
        if pa[2] <= 0.2 or pb[2] <= 0.2:
            continue
        d.line([tuple(pa[:2] / pa[2]), tuple(pb[:2] / pb[2])], fill=col, width=4)
    return np.asarray(img, dtype=np.uint8), arcs


def ground_truth_H(rx, ry, rz, w=W, h=H, f=F):
    """Normalised homography that rectifies the image (pixel frame, [0,1] coords)."""
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
    N = np.diag([1.0 / w, 1.0 / h, 1.0])
    Hm = N @ K @ rot(rx, ry, rz) @ np.linalg.inv(K) @ np.linalg.inv(N)
    return Hm / Hm[2, 2]


def main_nm():
    os.makedirs(NM, exist_ok=True)
    for name in SCENES:
        a, arcs = render_scene(name)
        Image.fromarray(a).save(os.path.join(NM, name + ".tif"),
                                compression="tiff_lzw")
        print("wrote", name, "arcs:", {k: v.shape for k, v in arcs.items()})
