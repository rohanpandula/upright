"""
Synthetic validation + demo for the CVPR-2012 upright energy.

Renders a 3-family (Manhattan) building corner under a known tilted camera,
then compares:
  (a) input                       - tilted, converging verticals
  (b) naive full rectification     - both horizontal VPs forced to infinity
  (c) paper's 'auto' upright       - adaptive lambda, perceptual energy
  (d) paper's 'full' mode          - aggressive

The point of the demo: (b) and (c) are geometrically different, and (c) is the
thing Lightroom actually does.
"""

import sys
from functools import partial

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from upright_core import (  # noqa: E402
    _normalize, detect_lines, estimate_vanishing_points, classify_families,
    energy, homography_from_params, intrinsics, warp, constrain_crop,
    line_homogeneous, _edge_points, crop_matrix_of,
)
from scipy.optimize import minimize  # noqa: E402

W, H_IMG = 1200, 900
F = 1400.0
RX, RY, RZ = -0.38, 0.10, 0.04   # pitch up, yaw right, slight roll


def rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def world_segments():
    segs = []
    # wall A: plane z = 25  (x-family horizontals + verticals)
    for x in np.arange(-8, 8.1, 2.0):
        segs.append(((x, -5, 25), (x, 18, 25)))
    for y in np.arange(-5, 18.1, 1.5):
        segs.append(((-8, y, 25), (8, y, 25)))
    # wall B: plane x = 8  (z-family receding horizontals + verticals)
    for z in np.arange(25, 75.1, 2.5):
        segs.append(((8, -5, z), (8, 18, z)))
    for y in np.arange(-5, 18.1, 1.5):
        segs.append(((8, y, 25), (8, y, 75)))
    return segs


def render(segs, rx=RX, ry=RY, rz=RZ, w=W, h=H_IMG, f=F):
    img = Image.new("L", (w, h), 245)
    d = ImageDraw.Draw(img)
    K = intrinsics(f, f, w / 2.0, h / 2.0)
    R = rot(rx, ry, rz)
    for a, b in segs:
        pa = K @ (R.T @ np.array(a, dtype=float))
        pb = K @ (R.T @ np.array(b, dtype=float))
        if pa[2] <= 0.2 or pb[2] <= 0.2:
            continue
        d.line([tuple(pa[:2] / pa[2]), tuple(pb[:2] / pb[2])], fill=30, width=2)
    return np.asarray(img, dtype=np.uint8)


def naive_full_rectification(vp_x, vp_y, vp_z, K):
    """Force all three Manhattan directions onto the camera axes."""
    K_inv = np.linalg.inv(K)
    dx = _normalize(K_inv @ vp_x)
    dy = _normalize(K_inv @ vp_y)
    dz = _normalize(K_inv @ vp_z)
    if dy[1] < 0:
        dy = -dy
    # Gram-Schmidt, gravity first
    e1 = dy
    e2 = dx - e1 * (e1 @ dx)
    e2 = _normalize(e2)
    e3 = _normalize(np.cross(e1, e2))
    Rm = np.stack([e2, e1, e3], axis=0)      # rows = world axes in camera coords
    return K @ Rm @ K_inv


def _diag(H, segs, vertical=True):
    """Mean residual tilt (deg) after correction + local stretch stats."""
    l = np.array([np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0]) for s in segs])
    lt = (np.linalg.inv(H).T @ l.T).T
    d = np.stack([-lt[:, 1], lt[:, 0]], axis=1)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    tilt = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1]))) if vertical \
        else np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    yy, xx = np.mgrid[0:H_IMG:8, 0:W:8]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    q = (H @ p.T).T
    Wc = np.where(np.abs(q[:, 2]) < 1e-9, 1e-9, q[:, 2])
    X, Y = q[:, 0] / Wc, q[:, 1] / Wc
    J = np.empty((len(p), 2, 2))
    J[:, 0, 0] = (H[0, 0] - X * H[2, 0]) / Wc
    J[:, 0, 1] = (H[0, 1] - X * H[2, 1]) / Wc
    J[:, 1, 0] = (H[1, 0] - Y * H[2, 0]) / Wc
    J[:, 1, 1] = (H[1, 1] - Y * H[2, 1]) / Wc
    a, b, c, d_ = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    tr = a * a + b * b + c * c + d_ * d_
    det = a * d_ - b * c
    disc = np.sqrt(np.maximum(tr * tr - 4 * det * det, 0.0))
    s1 = np.sqrt(np.maximum((tr + disc) / 2, 1e-12))
    s2 = np.sqrt(np.maximum((tr - disc) / 2, 1e-12))
    st = s1 / s2
    return float(np.mean(tilt)), float(np.percentile(st, 99)), float(st.max())


def main():
    segs3d = world_segments()
    gray = render(segs3d)
    Image.fromarray(gray).save("01_input.png")

    lines = detect_lines(gray)
    vps = estimate_vanishing_points(lines, W, H_IMG)
    fam = classify_families(vps, lines)
    print(f"detected {len(lines)} segments, {len(vps)} vanishing points")
    for v in vps:
        print(f"   vp={np.round(v['vp'], 3)} inliers={len(v['inliers'])} "
              f"support={v['weight']:.0f}")

    vert, horiz = fam["vertical"], fam["horizontals"]
    if vert is None or len(horiz) < 2:
        print("family detection failed; got", fam.keys(), len(horiz))
        return

    K = intrinsics(F, F, W / 2.0, H_IMG / 2.0)

    # curved/edge content drives E_img; wall/ground junctions drive Perkins
    edge_pts = _edge_points(gray, W)
    from upright_core import _junctions
    junctions = _junctions(lines[vert["inliers"]], lines[horiz[0]["inliers"]], W, H_IMG)
    print(f"edge points={len(edge_pts)} junctions={len(junctions)}")

    # ---- reference: naive full rectification -------------------------
    H_naive = naive_full_rectification(horiz[0]["vp"], vert["vp"], horiz[1]["vp"], K)

    # ---- our energy, auto and full ------------------------------------
    panel = [("input", gray, np.eye(3))]
    panel.append(("naive full rectification", gray, H_naive))

    for mode in ("auto", "full", "level", "vertical"):
        x0 = np.array([F, F, 0.0, 0.0, 0.0, 0.0, 0.0])
        K_inv = np.linalg.inv(K)
        dv = _normalize(K_inv @ vert["vp"])
        if dv[2] < 0:
            dv = -dv
        pitch0 = float(np.arctan2(-dv[2], abs(dv[1]) + 1e-12))
        dh = _normalize(K_inv @ horiz[0]["vp"])
        if dh[2] < 0:
            dh = -dh
        yaw0 = float(np.arctan2(dh[0], abs(dh[2]) + 1e-12))
        lam_v = float(np.exp(-(pitch0 ** 2) / (2 * (np.pi / 12) ** 2)))
        lam_h = float(np.exp(-(yaw0 ** 2) / (2 * (np.pi / 15) ** 2)))
        cfg = dict(w_pic=1.0, w_eye=1.0, w_img=3.0, w_f=30.0)
        if mode == "level":
            lam_v, lam_h = 0.10, 4.0
        elif mode == "vertical":
            lam_v, lam_h = 4.0, 0.10
        elif mode == "full":
            # aggressive: strong picture-frame pull, weak distortion regulariser
            lam_v = lam_h = 12.0
            cfg["w_img"] = 0.15
            cfg["w_f"] = 2.0

        res = minimize(
            partial(
                energy, K0=K, vert_lines=lines[vert["inliers"]],
                horiz_lines=lines[horiz[0]["inliers"]], vp_vert=vert["vp"],
                vp_h_x=horiz[0]["vp"], vp_h_z=horiz[1]["vp"], f0=F,
                edge_pts=edge_pts, junctions=junctions,
                lam_v=lam_v, lam_h=lam_h, **cfg,
            ),
            x0, method="Powell",
            options=dict(maxiter=1500, xtol=1e-6, ftol=1e-10),
        )
        Hm = homography_from_params(res.x, K)
        panel.append((f"energy: {mode}  (E={res.fun:.3f})", gray, Hm))
        print(f"[{mode}] E={res.fun:.5f} lam=({lam_v:.2f},{lam_h:.2f}) "
              f"f1/f0={res.x[0]/F:.4f} rx={res.x[2]:.4f} ry={res.x[3]:.4f} rz={res.x[4]:.4f}")

    # ---- metrics: residual convergence vs distortion vs crop loss ------
    print("\n  panel                        vert_tilt  horiz_tilt  p99_stretch  crop_keep")
    for label, _src, Hm in panel:
        vt, _, _ = _diag(Hm, lines[vert["inliers"]], True)
        ht, _, _ = _diag(Hm, lines[horiz[0]["inliers"]], False)
        _, p99, _ = _diag(Hm, lines[vert["inliers"]], True)
        crop, _c = constrain_crop(Hm, W, H_IMG)

        print(f"  {label:<28} {vt:8.3f}° {ht:10.3f}° {p99:11.3f} {crop*100:9.1f}%")

    # ---- render comparison, cropped to the valid region ---------------
    tiles = []
    for label, src, Hm in panel:
        sc, ctr = constrain_crop(Hm, W, H_IMG)
        Hc = crop_matrix_of(sc, ctr, W, H_IMG) @ Hm
        out = warp(src, Hc)
        out = np.clip(out, 0, 255).astype(np.uint8)
        tiles.append((label, out))
        safe = label.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        Image.fromarray(out).save(f"out_{safe}.png")

    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("L", (cols * W, rows * (H_IMG + 26)), 255)
    dr = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(Image.fromarray(im), (c * W, r * (H_IMG + 26) + 26))
        dr.text((c * W + 8, r * (H_IMG + 26) + 7), label, fill=0)
    sheet.save("comparison.png")
    print("wrote comparison.png")


if __name__ == "__main__":
    main()
