"""
Measure Adobe Upright on non-Manhattan scene families (and Manhattan controls).

The stored UprightTransform is not exactly K R K^-1 -- it carries its own
translation/aspect normalisation -- so comparing raw matrix elements is
meaningless. Instead every metric here is geometric and convention free:

  mag        max|H-I|, H rescaled to H[2,2]=1 (how hard did it correct at all)
  f35        UprightFocalLength35mm it solved with (truth for 1400 px: 41.14)
  vdev       median |angle from vertical| of the known vertical 3-D lines, px
  hdev       the same against horizontal
  arc res    circle-fit RMS (px) of a known arc/ring, before -> after
  aff_res    max px distance between the Adobe warp and the ground-truth
             rectification after fitting the best affine map. A correct
             rotation recovery differs from truth only by an affine map, so this
             isolates shape error. ~0 = perfect, hundreds = wrong warp.
  fill%      how much of the output frame is still covered by real image after
             the warp (a destructive over-crop shows up here)

Scene geometry is re-generated from make_test_images, so ground truth is exact.
"""

import json
import os
import re
import sqlite3
import sys

import numpy as np

import make_test_images as M

DB = "/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat"
HERE = os.path.dirname(os.path.abspath(__file__))
W, H, F = M.W, M.H, M.F
MARGIN = 0.0
GT_F35 = F * 36.0 / W          # 41.14 mm

RE_T = re.compile(r'UprightTransform_(\d)\s*=\s*"([^"]*)"')
RE_F = re.compile(r'UprightFocalLength35mm\s*=\s*([\d.]+)')


def pix_to_norm(Hn):
    N = np.diag([1.0 / W, 1.0 / H, 1.0])
    return np.linalg.inv(N) @ Hn @ N


def warp(Hn, pts):
    Hp = pix_to_norm(Hn)
    q = (Hp @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    return q[:, :2] / q[:, 2:3]


def project(pts3, R):
    K = np.array([[F, 0, W / 2.0], [0, F, H / 2.0], [0, 0, 1]])
    q = (K @ (R.T @ np.asarray(pts3, float).T)).T
    return q[:, :2] / q[:, 2:3]


def line_dev(pts, axis):
    d = np.diff(pts, axis=0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    dev = np.abs(np.abs(ang) - 90.0) if axis == 0 else np.minimum(
        np.abs(ang), 180.0 - np.abs(ang))
    dev = dev[np.abs(dev) < 80.0]
    return float(np.median(dev)) if len(dev) else None


def decompose(Hm, f_px):
    """Recover (pitch, yaw, roll) deg from a normalised rectifying homography.

    Only the first two columns of K^-1 H K are used: they are exactly the
    corresponding columns of R (scaled by 1/f), whereas the third column is
    polluted by H's own translation, which would bias an SVD projection."""
    K = np.array([[f_px, 0, W / 2.0], [0, f_px, H / 2.0], [0, 0, 1]])
    A = np.linalg.inv(K) @ pix_to_norm(Hm) @ K
    if abs(A[2, 2]) < 1e-12:
        return None
    A = A / A[2, 2]
    c0 = A[:, 0].copy()
    c1 = A[:, 1].copy()
    if np.linalg.norm(c0) < 1e-9 or np.linalg.norm(c1) < 1e-9:
        return None
    c2 = np.cross(c0, c1)
    Rm = np.column_stack([c0 / np.linalg.norm(c0),
                          c1 / np.linalg.norm(c1),
                          c2 / np.linalg.norm(c2)])
    U, _, Vt = np.linalg.svd(Rm)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        Rm = U @ np.diag([1, 1, -1]) @ Vt
    if abs(Rm[2, 0]) > 0.999:
        return None
    return (np.degrees(np.arctan2(Rm[2, 1], Rm[2, 2])),
            np.degrees(np.arcsin(-Rm[2, 0])),
            np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])))


def vp_point(lines):
    """Least-squares common intersection point of a set of lines."""
    if len(lines) < 3:
        return None
    A_ = np.zeros((2, 2))
    b_ = np.zeros(2)
    for pts in lines:
        d = np.diff(pts, axis=0).mean(0)
        n = np.array([-d[1], d[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        mid = pts[len(pts) // 2]
        A_ += np.outer(n, n)
        b_ += n * (n @ mid)
    try:
        return np.linalg.solve(A_ + 1e-9 * np.eye(2), b_)
    except np.linalg.LinAlgError:
        return None


def converge_dev(lines):
    """Median angle (deg) between each line and the ray from the lines' common
    vanishing point to its midpoint. 0 = all depth lines are concurrent, which
    is exactly what rectifying an optical-axis direction does. Translation
    invariant, so a recentring warp does not bias it."""
    if len(lines) < 3:
        return None
    A_ = np.zeros((2, 2))
    b_ = np.zeros(2)
    dirs, mids = [], []
    for pts in lines:
        d = np.diff(pts, axis=0).mean(0)
        n = np.array([-d[1], d[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        mid = pts[len(pts) // 2]
        A_ += np.outer(n, n)
        b_ += n * (n @ mid)
        dirs.append(d / np.linalg.norm(d))
        mids.append(mid)
    if len(dirs) < 3:
        return None
    P = np.linalg.solve(A_ + 1e-6 * np.eye(2), b_)
    devs = []
    for d, mid in zip(dirs, mids):
        ray = mid - P
        if np.linalg.norm(ray) < 1e-6:
            continue
        c = np.clip(abs(np.dot(d, ray / np.linalg.norm(ray))), 0.0, 1.0)
        devs.append(np.degrees(np.arccos(c)))
    return float(np.median(devs)) if devs else None


def circle_fit(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return float(r), float(np.sqrt(np.mean(
        (np.hypot(x - cx, y - cy) - r) ** 2)))


def ellipse_fit(pts):
    """Fitzgibbon conic fit -> (geometric rms px, major/minor axis ratio).
    A correctly rectified in-plane circle has ratio 1.0."""
    x, y = pts[:, 0], pts[:, 1]
    D = np.c_[x * x, x * y, y * y, x, y, np.ones(len(x))]
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    a, b, c, d, e, f = Vt[-1]
    Mx = np.array([[a, b / 2], [b / 2, c]])
    if np.linalg.det(Mx) <= 0:
        return None
    cen = np.linalg.solve(2 * Mx, -np.array([d, e]))
    val = cen @ Mx @ cen - f
    ev, evec = np.linalg.eigh(Mx)
    if np.any(val / ev <= 0):
        return None
    ax = np.sqrt(val / ev)
    t = np.linspace(0, 2 * np.pi, 720)
    samp = (evec @ np.array([ax[0] * np.cos(t), ax[1] * np.sin(t)])).T + cen
    d2 = ((pts[:, None, :] - samp[None, :, :]) ** 2).sum(-1).min(1)
    return float(np.sqrt(d2.mean())), float(ax.max() / ax.min())


def sample_line(a, b, n=24):
    t = np.linspace(0.08, 0.92, n)[:, None]
    return np.asarray(a, float) + t * (np.asarray(b, float) - np.asarray(a, float))


def affine_residual(Ha, Hb, pts):
    """Best-affine-aligned max px distance between two warps, and the anisotropy
    of the fitted linear part (1.0 = no aspect distortion)."""
    qa, qb = warp(Ha, pts), warp(Hb, pts)
    ok = np.all(np.isfinite(qa), axis=1) & np.all(np.isfinite(qb), axis=1)
    qa, qb = qa[ok], qb[ok]
    if len(qa) < 12:
        return None, None
    X = np.c_[qa, np.ones(len(qa))]
    sol, *_ = np.linalg.lstsq(X, qb, rcond=None)
    res = float(np.max(np.abs(X @ sol - qb)))
    sv = np.linalg.svd(sol[:2, :2], compute_uv=False)
    return res, float(sv[0] / max(sv[1], 1e-9))


def sim_residual(Ha, Hb, pts):
    """Best-similarity-aligned max px distance between two warps. Unlike the
    affine version this keeps aspect distortion, so a wrong focal shows up."""
    qa, qb = warp(Ha, pts), warp(Hb, pts)
    ok = np.all(np.isfinite(qa), axis=1) & np.all(np.isfinite(qb), axis=1)
    qa, qb = qa[ok], qb[ok]
    if len(qa) < 12:
        return None, None
    ca, cb = qa.mean(0), qb.mean(0)
    z = (qa - ca)[:, 0] + 1j * (qa - ca)[:, 1]
    w = (qb - cb)[:, 0] + 1j * (qb - cb)[:, 1]
    c = np.vdot(z, w) / np.vdot(z, z)
    res = np.abs(c * z + (cb[0] + 1j * cb[1]) - (w + (cb[0] + 1j * cb[1])))
    return float(np.max(res)), float(abs(c))


def clip_area(quad):
    """Area of the intersection of a convex quad with the frame rectangle."""
    def clip(poly, axis, val, keep_greater):
        out = []
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            pin = (p[axis] >= val) if keep_greater else (p[axis] <= val)
            qin = (q[axis] >= val) if keep_greater else (q[axis] <= val)
            if pin:
                out.append(p)
            if pin != qin:
                t = (val - p[axis]) / (q[axis] - p[axis])
                out.append(p + t * (q - p))
        return out
    poly = [np.array(p, float) for p in quad]
    for axis, val, gr in ((0, 0.0, True), (0, W, False), (1, 0.0, True), (1, H, False)):
        if not poly:
            return 0.0
        poly = clip(poly, axis, val, gr)
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def frame_fill(Hn):
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    q = warp(Hn, corners)
    if not np.all(np.isfinite(q)):
        return 0.0
    quad_area = 0.0
    for i in range(4):
        x1, y1 = q[i]
        x2, y2 = q[(i + 1) % 4]
        quad_area += x1 * y2 - x2 * y1
    quad_area = abs(quad_area) / 2.0
    if quad_area < 1.0:
        return 0.0
    # fraction of the *source* pixels that still land inside the output frame
    return 100.0 * clip_area(q) / quad_area


def stored(image_id, con):
    row = con.execute("SELECT text FROM Adobe_imageDevelopSettings WHERE image=?",
                      (image_id,)).fetchone()
    if not row:
        return None
    tr = {}
    for n, v in RE_T.findall(row[0]):
        nums = [float(x) for x in v.split(",")]
        if len(nums) == 9:
            tr[int(n)] = np.array(nums).reshape(3, 3)
    f = RE_F.search(row[0])
    return tr, (float(f.group(1)) if f else None)


def photo_id(base, con):
    return con.execute(
        "SELECT i.id_local FROM Adobe_images i JOIN AgLibraryFile f "
        "ON f.id_local=i.rootFile WHERE f.baseName=? ORDER BY i.id_local DESC "
        "LIMIT 1", (base,)).fetchone()


ROWS = [(n, M.PITCH15, M.YAW6) for n in M.SCENES] + [
    ("A_clean", -0.384, 0.105),      # Manhattan control, pitch 22 / yaw 6
    ("B_strong", -0.60, 0.21),       # Manhattan control, pitch 34 / yaw 12
]


def verdict(r):
    if r["mag"] < 0.01:
        return "DEGENERATE"
    if (r["sim"] is not None and r["sim"] > 60) or (r["vdev_post"] or 99) > 3 \
            or (r["hdev_post"] or 99) > 3 or r["fill"] < 40:
        return "WRONG"
    if r.get("arc_ratio") and r["arc_ratio"] > 2.0:
        return "WRONG (bends arc)"
    if r["sim"] is not None and r["sim"] < 25 and r["fill"] >= 55:
        return "sensible"
    return "marginal"


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = {}
    for name, rx, ry in ROWS:
        R = M.rot(rx, ry, 0.0)
        Hgt = M.ground_truth_H(rx, ry, 0.0)
        Hgtp = M.ground_truth_H(rx, 0.0, 0.0)
        if name in M.SCENES:
            segs, arcs = M.SCENES[name]()
        else:                       # Manhattan control: derive the families
            segs, arcs = [], {}
            for a, b in M.world_segments():
                d = np.subtract(b, a)
                if abs(d[0]) < 1e-9 and abs(d[2]) < 1e-9:
                    fam = "v"
                elif abs(d[1]) < 1e-9 and abs(d[2]) < 1e-9:
                    fam = "h"
                else:
                    fam = "d"
                segs.append((a, b, 0, fam))
        row = photo_id(name, con)
        rec = {"present": bool(row), "gt_pitch": round(-np.degrees(rx), 2),
               "gt_yaw": round(np.degrees(ry), 2)}
        if row:
            s = stored(row[0], con)
            if s:
                tr, f35 = s
                for mode, slot in (("auto", 1), ("full", 2)):
                    if slot not in tr or abs(tr[slot][2, 2]) < 1e-12:
                        continue
                    Hm = tr[slot] / tr[slot][2, 2]
                    r = {"mag": float(np.max(np.abs(Hm - np.eye(3)))),
                         "f35": f35}
                    grid = np.array([[(u, v) for u in np.linspace(60, W - 60, 9)]
                                     for v in np.linspace(60, H - 60, 7)],
                                    dtype=float).reshape(-1, 2)
                    both = {}
                    for tag, G in (("full", Hgt), ("pitch_only", Hgtp)):
                        sim, sc = sim_residual(Hm, G, grid)
                        aff, an = affine_residual(Hm, G, grid)
                        if sim is not None:
                            both[tag] = (sim, aff, an, sc)
                    if both:
                        tag = min(both, key=lambda k: both[k][0])
                        r["sim"], r["aff"], r["aniso"], r["gt_scale"] = both[tag]
                        r["matched"] = tag
                        base, _ = sim_residual(np.eye(3), Hgt if tag == "full" else Hgtp, grid)
                        r["base_sim"] = base
                        r["frac_fixed"] = round(1.0 - r["sim"] / base, 3) if base else None
                    else:
                        r["sim"] = r["aff"] = r["aniso"] = r["matched"] = None
                    r["fill"] = frame_fill(Hm)
                    for fam, key in (("v", "vdev"), ("h", "hdev"), ("d", "ddev")):
                        pre, post, lpre, lpost = [], [], [], []
                        for A_, B_, col, f in segs:
                            if f != fam:
                                continue
                            p = project(sample_line(A_, B_), R)
                            for Hw, acc in ((None, pre), (Hm, post)):
                                q = warp(Hw, p) if Hw is not None else p
                                if not np.all(np.isfinite(q)):
                                    continue
                                keep = ((q[:, 0] > -MARGIN) & (q[:, 0] < W + MARGIN)
                                        & (q[:, 1] > -MARGIN) & (q[:, 1] < H + MARGIN))
                                if keep.sum() < 8:
                                    continue
                                if fam == "d":
                                    acc.append(q[keep])
                                else:
                                    d = line_dev(q[keep], 0 if fam == "v" else 1)
                                    if d is not None:
                                        acc.append(d)
                        if fam == "d":
                            r[key + "_pre"] = converge_dev(pre)
                            r[key + "_post"] = converge_dev(post)
                            vpre_, vpost_ = vp_point(pre), vp_point(post)
                            if vpre_ is not None and vpost_ is not None:
                                r["vp_dist_pre"] = float(np.hypot(
                                    vpre_[0] - W / 2, vpre_[1] - H / 2))
                                r["vp_dist_post"] = float(np.hypot(
                                    vpost_[0] - W / 2, vpost_[1] - H / 2))
                        else:
                            r[key + "_pre"] = float(np.median(pre)) if pre else None
                            r[key + "_post"] = float(np.median(post)) if post else None
                        r[key + "_n"] = len(post)
                    # residual rotation against truth (positive = still uncorrected)
                    if f35:
                        try:
                            Hres = Hgt @ np.linalg.inv(Hm)
                            rot = decompose(Hres, f35 * W / 36.0)
                        except np.linalg.LinAlgError:
                            rot = None
                        if rot:
                            r["res_pitch"], r["res_yaw"], r["res_roll"] = (
                                round(rot[0], 2), round(rot[1], 2), round(rot[2], 2))
                    if arcs:
                        for lab, p3 in arcs.items():
                            p = project(p3, R)
                            q = warp(Hm, p)
                            r0, e0 = circle_fit(p)
                            r1, e1 = circle_fit(q)
                            ef = ellipse_fit(q)
                            r.setdefault("arc", {})[lab] = {
                                "res_pre": round(e0, 3), "res_post": round(e1, 3),
                                "r_pre": round(r0, 1), "r_post": round(r1, 1),
                                "aspect_post": round(ef[1], 4) if ef else None,
                                "ellipse_rms": round(ef[0], 3) if ef else None}
                        es = [v["res_post"] for v in r["arc"].values()]
                        e0s = [v["res_pre"] for v in r["arc"].values()]
                        r["arc_ratio"] = round(float(np.median(es) /
                                                     np.median(e0s)), 3)
                    r["verdict"] = verdict(r)
                    rec[mode] = r
        out[name] = rec
    print(json.dumps(out, indent=1, default=str))
    with open(os.path.join(HERE, "nm_metrics.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nscene,mode,mag,f35,vdev_pre,vdev_post,hdev_pre,hdev_post,sim_px,base_px,aniso,fill%,arc_ratio,verdict,zvp_dist,res_pitch,res_yaw,res_roll")
    for n, rec in out.items():
        for mode in ("auto", "full"):
            r = rec.get(mode)
            if not r:
                print(f"{n},{mode},MISSING"); continue
            print("%s,%s,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s,%s,%.1f,%s,%s,%s" % (
                n, mode, r["mag"], r["f35"] or -1, r["vdev_pre"] or -1,
                r["vdev_post"] or -1, r["hdev_pre"] or -1, r["hdev_post"] or -1,
                ("%.1f" % r["sim"]) if r["sim"] is not None else "-",
                ("%.1f" % r["base_sim"]) if r.get("base_sim") else "-",
                ("%.2f" % r["aniso"]) if r["aniso"] else "-", r["fill"],
                r.get("arc_ratio", "-"), r["verdict"],
                ("%.0f->%.0f" % (r["vp_dist_pre"], r["vp_dist_post"]))
                if r.get("vp_dist_pre") is not None else "-") +
                (",%s,%s,%s" % (r.get("res_pitch"), r.get("res_yaw"), r.get("res_roll"))))


if __name__ == "__main__":
    sys.exit(main())
