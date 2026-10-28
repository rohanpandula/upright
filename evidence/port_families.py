"""Port stress test on families and poses it was not tuned for.

Scores the port exactly like the Manhattan scenes are scored in
port/benchmark_accuracy.py: ground-truth world lines are projected through the
known generating rotation, the port's H is applied, and the median residual
axis deviation in degrees is reported, plus a best-similarity px residual
against the generator's ground-truth rectification H.  Also reports the gate
decision (ok/reason) so a gate failure is never confused with a geometric one.

Fixtures: make_test_images.render_scene / ground_truth_H.  MIT.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "port_snapshot"))  # pinned f2a900d
sys.path.insert(0, ROOT)

import make_test_images as M  # noqa: E402
import upright_port as up  # noqa: E402

W, H, F = M.W, M.H, M.F
FOCAL = F
MARGIN = 6
KIND = {"v": 0, "h": 1, "d": 2}


def project(pts3, R, f=FOCAL):
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]])
    q = (K @ (R.T @ np.asarray(pts3, float).T)).T
    return q[:, :2] / q[:, 2:3]


def pix_to_norm(Hn):
    N = np.diag([1.0 / W, 1.0 / H, 1.0])
    return np.linalg.inv(N) @ Hn @ N


def warp(Hn, pts):
    Hp = pix_to_norm(Hn)
    q = (Hp @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    return q[:, :2] / q[:, 2:3]


def visible_lines(segs, R, f=FOCAL):
    """Visible ground-truth lines: sampled in-frame points and their family."""
    out = []
    for a, b, col, fam in segs:
        p = project(np.array([a, b], float), R, f)
        if not np.all(np.isfinite(p)):
            continue
        t = np.linspace(0.02, 0.98, 25)[:, None]
        s = (1 - t) * p[0] + t * p[1]
        keep = np.all((s >= MARGIN) & (s < [W - MARGIN, H - MARGIN]), axis=1)
        q = s[keep]
        if len(q) < 8 or np.hypot(*(q[-1] - q[0])) < 40:
            continue
        out.append((q, KIND[fam]))
    return out


def deviation(Hn, lines):
    """Median axis deviation in deg, measured on each line's chord (the same
    chord angle benchmark_accuracy.deviation scores)."""
    Hp = pix_to_norm(Hn)
    res = {0: [], 1: []}
    for pts, kind in lines:
        if kind == 2:
            continue
        seg = np.array([[pts[0, 0], pts[0, 1], pts[-1, 0], pts[-1, 1]]])
        l = up._homogeneous_lines(seg) @ np.linalg.inv(Hp)
        ang = np.degrees(np.arctan2(np.abs(l[:, 0]), np.abs(l[:, 1])))
        dev = 90.0 - ang if kind == 0 else ang
        res[kind].append(float(np.median(dev)))
    return {k: (float(np.median(v)) if v else None) for k, v in res.items()}


def converge_dev(lines):
    """Median deg between each depth line and the ray from its common VP."""
    pts_list = [p for p, k in lines if k == 2]
    if len(pts_list) < 3:
        return None
    A_ = np.zeros((2, 2))
    b_ = np.zeros(2)
    dirs, mids = [], []
    for pts in pts_list:
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


def grid_pts():
    return np.array([[(u, v) for u in np.linspace(60, W - 60, 9)]
                     for v in np.linspace(60, H - 60, 7)],
                    dtype=float).reshape(-1, 2)


def sim_residual(Ha, Hb, pts):
    qa, qb = warp(Ha, pts), warp(Hb, pts)
    ok = np.all(np.isfinite(qa), axis=1) & np.all(np.isfinite(qb), axis=1)
    qa, qb = qa[ok], qb[ok]
    if len(qa) < 12:
        return None
    ca, cb = qa.mean(0), qb.mean(0)
    z = (qa - ca)[:, 0] + 1j * (qa - ca)[:, 1]
    w = (qb - cb)[:, 0] + 1j * (qb - cb)[:, 1]
    c = np.vdot(z, w) / np.vdot(z, z)
    return float(np.max(np.abs(c * z - w)))


def fill_pct(Hn):
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    q = warp(Hn, corners)
    if not np.all(np.isfinite(q)):
        return 0.0

    def clip(poly, axis, val, greater):
        out = []
        for i in range(len(poly)):
            p, r = poly[i], poly[(i + 1) % len(poly)]
            pin = (p[axis] >= val) if greater else (p[axis] <= val)
            rin = (r[axis] >= val) if greater else (r[axis] <= val)
            if pin:
                out.append(p)
            if pin != rin:
                t = (val - p[axis]) / (r[axis] - p[axis])
                out.append(p + t * (r - p))
        return out

    poly = [np.array(p, float) for p in q]
    area = 0.0
    for i in range(4):
        x1, y1 = q[i]
        x2, y2 = q[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2
    if area < 1:
        return 0.0
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
    return 100.0 * abs(s) / 2.0 / area


def decompose(Hm, f_px):
    K = np.array([[f_px, 0, W / 2.0], [0, f_px, H / 2.0], [0, 0, 1]])
    A = np.linalg.inv(K) @ pix_to_norm(Hm) @ K
    if abs(A[2, 2]) < 1e-12:
        return None
    A = A / A[2, 2]
    c0, c1 = A[:, 0], A[:, 1]
    if np.linalg.norm(c0) < 1e-9 or np.linalg.norm(c1) < 1e-9:
        return None
    c2 = np.cross(c0, c1)
    Rm = np.column_stack([c0 / np.linalg.norm(c0), c1 / np.linalg.norm(c1),
                          c2 / np.linalg.norm(c2)])
    U, _, Vt = np.linalg.svd(Rm)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        Rm = U @ np.diag([1, 1, -1]) @ Vt
    if abs(Rm[2, 0]) > 0.999:
        return None
    return (float(np.degrees(np.arctan2(Rm[2, 1], Rm[2, 2]))),
            float(np.degrees(np.arcsin(-Rm[2, 0]))),
            float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))))


def circle_fit(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return float(r), float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))


def ellipse_aspect(pts):
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
    return float(ax.max() / ax.min()), float(np.sqrt(d2.mean()))


def gate_class(reason):
    if not reason:
        return ""
    if "poor undamped" in reason:
        return "GATE"
    if "insufficient line structure" in reason:
        return "INPUT"
    if reason.startswith("only "):
        return "INPUT"
    if reason in ("bad focal", "non-finite homography"):
        return "INPUT"
    return reason


def score(out, lines, Hgt, Hgtp, base_cache, arcs, R, f_px):
    Hn = out["H"]
    rec = {"ok": out["ok"], "reason": out["reason"],
           "gate_class": gate_class(out["reason"]), "mag": float(np.max(np.abs(Hn - np.eye(3)))),
           "n_seg": int(len(out["segments"])),
           "solved_f35": round(up.focal_35mm_from_px(out["source_focal_px"], W, H), 2)
           if out.get("source_focal_px") else None,
           "true_f35": round(up.focal_35mm_from_px(f_px, W, H), 2)}
    d = deviation(Hn, lines)
    rec["vdev"], rec["hdev"] = d[0], d[1]
    rec["ddev"] = converge_dev([(warp(Hn, p), k) for p, k in lines])
    pts = grid_pts()
    both = {}
    for tag, G in (("full", Hgt), ("pitch_only", Hgtp)):
        s = sim_residual(Hn, G, pts)
        if s is not None:
            both[tag] = s
    if both:
        tag = min(both, key=both.get)
        rec["matched"], rec["sim"] = tag, round(both[tag], 2)
        base = base_cache[tag]
        rec["base"] = round(base, 2)
        rec["frac"] = round(1.0 - both[tag] / base, 3) if base else None
        rec["res_rot"] = decompose(G if tag == "full" else Hgtp, out.get("source_focal_px") or f_px)
    rec["fill"] = round(fill_pct(Hn), 1)
    if out.get("quality"):
        rec["q_scatter"], rec["q_bias"] = (out["quality"]["scatter_passed"],
                                           out["quality"]["bias_passed"])
    if out.get("calibration"):
        rec["cal"] = bool(out["calibration"].get("accepted"))
    if arcs:
        arc = {}
        for lab, p3 in arcs.items():
            p = project(p3, R, f_px)
            keep = np.all((p > -1e6) & np.isfinite(p), axis=1)
            p = p[keep]
            if len(p) < 20:
                continue
            q = warp(Hn, p)
            if not np.all(np.isfinite(q)):
                continue
            r0, e0 = circle_fit(p)
            r1, e1 = circle_fit(q)
            asp = ellipse_aspect(q)
            arc[lab] = {"r_pre": round(r0, 1), "r_post": round(r1, 1),
                        "rms_pre": round(e0, 3), "rms_post": round(e1, 3),
                        "aspect": round(asp[0], 4) if asp else None}
        gt_arc = {}
        for lab, p3 in arcs.items():
            p = project(p3, R, f_px)
            p = p[np.all(np.isfinite(p), axis=1)]
            if len(p) < 20:
                continue
            q = warp(Hgt, p)
            if not np.all(np.isfinite(q)):
                continue
            r1, e1 = circle_fit(q)
            gt_arc[lab] = {"r_gt": round(r1, 1), "rms_gt": round(e1, 3)}
        rec["arcs"] = arc
        rec["arcs_gt"] = gt_arc
        if arc:
            rec["arc_ratio"] = round(float(np.median([v["rms_post"] for v in arc.values()])
                                           / max(np.median([v["rms_pre"] for v in arc.values()]), 1e-9)), 3)
    return rec


def candidate_H(out, f_px):
    if out.get("rotation") is None:
        return None
    K = up._intrinsics_norm(out.get("source_focal_px") or f_px, W, H)
    return K @ up._rotation(*out["rotation"]) @ np.linalg.inv(K)


def run_cell(family, pose, pose_label, mode, f_px=FOCAL, texture=None):
    rx, ry, rz = pose
    img, arcs = M.render_scene(family, rx, ry, rz, f=f_px)
    if texture:
        img = texture(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    segs = up.detect_segments(gray)
    out = up.solve_ex(segs, W, H, f_px, mode)
    R = M.rot(rx, ry, rz)
    segs_w, _ = M.SCENES[family]()
    lines = visible_lines(segs_w, R, f_px)
    Hgt = M.ground_truth_H(rx, ry, rz, f=f_px)
    Hgtp = M.ground_truth_H(rx, 0.0, rz, f=f_px)
    pts = grid_pts()
    base = {"full": sim_residual(np.eye(3), Hgt, pts),
            "pitch_only": sim_residual(np.eye(3), Hgtp, pts)}
    rec = {"family": family, "pose": pose_label, "mode": mode,
           "visible_lines": len(lines),
           "identity_on_fail": bool(not out["ok"] and np.array_equal(out["H"], np.eye(3)))}
    pre = deviation(np.eye(3), lines)
    rec["vdev_pre"], rec["hdev_pre"] = pre[0], pre[1]
    rec.update(score(out, lines, Hgt, Hgtp, base, arcs, R, f_px))
    try:
        segn = segs / np.array([W, H, W, H], dtype=float)
        _v, _h = up._classify(up.estimate_vanishing_points(segs, W, H), segn)
        rec["h_fams"] = len(_h)
        rec["h_inliers"] = [len(x["inliers"]) for x in _h]
        rec["h_vp_x"] = [round(float(x["vp"][0] / x["vp"][2]), 0)
                         if abs(x["vp"][2]) > 1e-9 else None for x in _h]
        rec["v_inliers"] = len(_v["inliers"]) if _v else 0
    except Exception as exc:
        rec["h_fams"] = None
    rec["rot"] = [round(float(x), 5) for x in out["rotation"]] if out.get("rotation") else None
    rec["damping"] = [round(float(x), 4) for x in out["damping"]] if out.get("damping") else None
    rec["keystone_deg"] = round(float(np.degrees(out["vertical_keystone_rad"])), 2) \
        if out.get("vertical_keystone_rad") is not None else None
    if out.get("calibration"):
        c = out["calibration"]
        rec["cal_accepted"] = bool(c.get("accepted"))
        rec["cal_est_px"] = round(float(c.get("estimated_px", 0)), 1)
        rec["cal_res"] = round(float(c.get("log_focal_resolution", 0)), 4)
    if not out["ok"]:
        Hc = candidate_H(out, f_px)
        if Hc is not None:
            cd = deviation(Hc, lines)
            rec["cand_vdev"], rec["cand_hdev"] = cd[0], cd[1]
            pts = grid_pts()
            both = {t: sim_residual(Hc, G, pts)
                    for t, G in (("full", Hgt), ("pitch_only", Hgtp))}
            both = {k: v for k, v in both.items() if v is not None}
            if both:
                tag = min(both, key=both.get)
                rec["cand_match"] = tag
                rec["cand_sim"] = round(both[tag], 2)
                rec["cand_frac"] = round(1.0 - both[tag] / base[tag], 3) if base[tag] else None
        if out.get("quality"):
            rec["bias_bound"] = [round(float(x), 3)
                                 for x in out["quality"]["family_bias_bound_deg"]]
    return rec


# --------------------------------------------------------------------------
# synthetic featureless / textured frames
# --------------------------------------------------------------------------
def gradient(w=W, h=H, seed=0):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u, v = xx / w, yy / h
    g = 90 + 70 * np.sqrt(np.clip((u - 0.25) ** 2 + (v - 0.2) ** 2, 0, 1.4))
    g += 25 * u + 10 * v
    return np.clip(g, 0, 255).astype(np.uint8)


def ramp(w=W, h=H):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    return np.clip(60 + 120 * xx / w + 40 * yy / h, 0, 255).astype(np.uint8)


def add_lines(img, n, pose=(M.PITCH15, M.YAW6, 0.0), seed=0, f=FOCAL):
    """n vertical + n horizontal world lines on the z=30 plane, projected."""
    rng = np.random.default_rng(seed)
    R = M.rot(*pose)
    rgb = np.stack([img] * 3, -1).astype(np.uint8)
    for i in range(n):
        x = rng.uniform(-10, 14)
        y = rng.uniform(-6, 18)
        a = (x, -8.0, 30.0)
        b = (x, 20.0, 30.0)
        if i % 2:
            a, b = (-14.0, y, 30.0), (18.0, y, 30.0)
        p = project(np.array([a, b], float), R, f)
        if np.all(np.isfinite(p)):
            cv2.line(rgb, tuple(np.int32(p[0])), tuple(np.int32(p[1])), (30, 30, 30), 4)
    return rgb


def heavy_texture(img, seed=1):
    rng = np.random.default_rng(seed)
    rgb = img.astype(np.int16)
    rgb += rng.normal(0, 34, rgb.shape).astype(np.int16)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    for _ in range(500):
        x, y = rng.integers(0, W), rng.integers(0, H)
        ang = rng.uniform(0, np.pi)
        ln = rng.uniform(15, 70)
        q = (int(x + ln * np.cos(ang)), int(y + ln * np.sin(ang)))
        col = int(rng.uniform(10, 90))
        cv2.line(rgb, (int(x), int(y)), q, (col, col, col), 2)
    return rgb


def main():
    rows = []
    poses = [
        ("base15/6", (M.PITCH15, M.YAW6, 0.0)),
        ("pitch40", (np.radians(-40), M.YAW6, 0.0)),
        ("pitch48", (np.radians(-48), M.YAW6, 0.0)),
        ("pitch55", (np.radians(-55), M.YAW6, 0.0)),
        ("yaw30", (M.PITCH15, np.radians(30), 0.0)),
        ("yaw35", (M.PITCH15, np.radians(35), 0.0)),
        ("yaw40", (M.PITCH15, np.radians(40), 0.0)),
        ("roll20", (M.PITCH15, M.YAW6, np.radians(20))),
        ("roll25", (M.PITCH15, M.YAW6, np.radians(25))),
        ("roll30", (M.PITCH15, M.YAW6, np.radians(30))),
        ("frontal2/1", (np.radians(-2), np.radians(1), 0.0)),
        ("frontal1/0.5", (np.radians(-1), np.radians(0.5), 0.0)),
    ]
    families = ["NM_single_wall", "NM_railing", "NM_curves", "NM_three_plane",
                "CT_manhattan2"]

    # Family block: fixture focal 1600, base pose + extreme poses at 900 px
    # (1600 px framing drops most content beyond 40 deg, see visible_lines).
    for label, pose in poses:
        focals = (FOCAL,) if label.startswith(("base", "frontal")) else (FOCAL, 900.0)
        for f_px in focals:
          for fam in families:
            for mode in ("auto", "full") + (("level",) if label.startswith("roll") else ()):
                rec = run_cell(fam, pose, label, mode, f_px=f_px)
                rec["focal"] = int(f_px)
                rows.append(rec)

    # Featureless / textured block at the base pose, focal 1600.
    g = gradient()
    r = ramp()
    noisy = np.clip(r + np.random.default_rng(0).normal(0, 8, r.shape), 0, 255).astype(np.uint8)
    for label, img in (
        ("flat_gradient_0line", g),
        ("ramp_0line", r),
        ("ramp_noise_0line", noisy),
        ("gradient_3v", add_lines(g, 3)),
        ("gradient_5mix", add_lines(g, 5)),
        ("ramp_5mix", add_lines(r, 5)),
        ("texture_only", heavy_texture(gradient())),
    ):
        segs = up.detect_segments(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                                  if img.ndim == 3 else img)
        out = up.solve_ex(segs, W, H, FOCAL, "auto")
        row = {"family": label, "pose": "base15/6", "mode": "auto",
               "ok": out["ok"], "reason": out["reason"],
               "gate_class": gate_class(out["reason"]),
               "n_seg": int(len(segs)),
               "identity_on_fail": bool(not out["ok"]
                                        and np.array_equal(out["H"], np.eye(3))),
               "mag": float(np.max(np.abs(out["H"] - np.eye(3)))),
               "solved_f35": round(up.focal_35mm_from_px(out["source_focal_px"], W, H), 2)
               if out.get("source_focal_px") else None}
        if out.get("rotation"):
            row["rot"] = [round(float(x), 3) for x in out["rotation"]]
            row["damping"] = [round(float(x), 4) for x in out["damping"]]
            Hc = candidate_H(out, FOCAL)
            row["cand_mag"] = float(np.max(np.abs(Hc - np.eye(3))))
        rows.append(row)

    # Textured real scene: does clutter degrade a solvable frame?
    for fam in ("NM_three_plane", "CT_manhattan2"):
        rows.append(run_cell(fam, (M.PITCH15, M.YAW6, 0.0), "base15/6", "full",
                             texture=heavy_texture))

    with open(os.path.join(HERE, "port_families.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
