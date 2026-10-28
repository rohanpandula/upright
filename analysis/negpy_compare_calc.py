"""Numerical comparison of NegPy's keystone_quad family against Lightroom's
s*K1*R*K0^-1, on a controlled pitch-15 / yaw-6 frame.

Fair-to-NegPy construction:
  * NegPy's OWN warp (negpy.features.geometry.logic.keystone_matrix);
  * NegPy's OWN residual definition (skew._line_angles: angle off family axis);
  * the fit is a free Gauss-Newton over its full 3-parameter model
    (fine_rotation + converge_v + converge_h) with NO cap, NO gain gate and NO
    confidence gate -- so "best-in-family" is the optimum NegPy's own solver
    would reach with its safety rails removed.

Only the families a gate edge would actually expose are fitted/scored:
verticals (y) and fronto-parallel horizontals (x). Depth lines (z) are excluded
because they radiate from the principal point and no gate edge is a depth line.
"""

import sys
import numpy as np

sys.path.insert(0, "/Users/rohan/Projects/negpy")
sys.path.insert(0, "/Users/rohan/Projects/trichroma/research/upright")

import negpy_core  # noqa: E402
from negpy.features.geometry.logic import keystone_matrix  # noqa: E402
from make_test_images import world_segments  # noqa: E402

W, H, F = 1400, 1000, 1600.0


def rot(rx, ry, rz=0.0):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


K = np.array([[F, 0, W / 2.0], [0, F, H / 2.0], [0, 0, 1]])


def clip_seg(p1, p2):
    x0, y0 = p1
    dx, dy = p2[0] - x0, p2[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, W - 1 - x0), (-dy, y0), (dy, H - 1 - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t1 - t0 < 1e-6:
        return None
    a = np.array([x0 + t0 * dx, y0 + t0 * dy])
    b = np.array([x0 + t1 * dx, y0 + t1 * dy])
    return None if np.hypot(*(b - a)) < 40 else (a, b)


def project(segs, R):
    out = []
    for a, b in segs:
        pa = K @ (R.T @ np.array(a, float))
        pb = K @ (R.T @ np.array(b, float))
        if pa[2] <= 0.2 or pb[2] <= 0.2:
            continue
        c = clip_seg(pa[:2] / pa[2], pb[:2] / pb[2])
        if c is None:
            continue
        d = np.array(b, float) - np.array(a, float)
        if abs(d[0]) < 1e-9 and abs(d[2]) < 1e-9:
            fam = "v"
        elif abs(d[1]) < 1e-9 and abs(d[2]) < 1e-9:
            fam = "hx"
        else:
            fam = "hz"
        out.append((c[0], c[1], fam))
    return out


def tilt(P, fam):
    """Signed angle off the family axis, deg, folded to (-90, 90] (skew._line_angles)."""
    d = P[:, 1] - P[:, 0]
    ang = np.where(fam == "v",
                   -np.degrees(np.arctan2(d[:, 0], d[:, 1])),
                   np.degrees(np.arctan2(d[:, 1], d[:, 0])))
    return (ang + 90.0) % 180.0 - 90.0


def apply_H(Hm, P):
    flat = P.reshape(-1, 2)
    hom = np.column_stack([flat, np.ones(len(flat))]) @ Hm.T
    return (hom[:, :2] / hom[:, 2:3]).reshape(-1, 2, 2)


def negpy_map(P, angle, cv, ch):
    r = negpy_core.rotation_matrix((W / 2.0, H / 2.0), angle, 1.0)
    flat = P.reshape(-1, 2)
    out = flat @ r[:, :2].T + r[:, 2]
    if abs(cv) > 1e-9 or abs(ch) > 1e-9:
        hom = np.column_stack([out, np.ones(len(out))]) @ keystone_matrix(cv, ch, W, H).T
        out = hom[:, :2] / hom[:, 2:3]
    return out.reshape(-1, 2, 2)


def metrics(P, fam):
    """(mean |vertical tilt|, mean |horizontal tilt|, vertical convergence angle)."""
    t = tilt(P, fam)
    a = np.abs(t)
    v, h = a[fam == "v"], a[fam == "hx"]
    conv = float(np.max(t[fam == "v"]) - np.min(t[fam == "v"])) if np.sum(fam == "v") >= 2 else float("nan")
    return float(np.mean(v)), float(np.mean(h)), conv


def fit_negpy(P, fam, cap=None, iters=200):
    """Gauss-Newton on NegPy's own objective, as skew._solve does."""
    keep = fam != "hz"
    P, fam = P[keep], fam[keep]
    w = np.sqrt(np.array([np.hypot(*(q[1] - q[0])) for q in P]))
    lim = cap if cap is not None else 60.0

    def res(x):
        return tilt(negpy_map(P, *x), fam) * w

    params = np.zeros(3)
    best, best_r = params.copy(), float(np.sqrt(np.mean(res(params) ** 2)))
    for _ in range(iters):
        r = res(params)
        jac = np.empty((len(P), 3))
        for j in range(3):
            st = np.zeros(3)
            st[j] = 1e-4
            jac[:, j] = (res(params + st) - r) / 1e-4
        delta, *_ = np.linalg.lstsq(jac, -r, rcond=None)
        params = params + delta
        params[1:] = np.clip(params[1:], -lim, lim)
        rr = float(np.sqrt(np.mean(res(params) ** 2)))
        if rr < best_r:
            best, best_r = params.copy(), rr
        if np.max(np.abs(delta)) < 1e-11:
            break
    a, cv, ch = best
    return float(a), float(np.clip(cv, -lim, lim)), float(np.clip(ch, -lim, lim))


def gate_quad(R, inset=0.06):
    """Source-image quad of a gate rectangle in the film plane (z=25).

    The frame corners are unprojected onto the film plane and their axis-aligned
    world bounding box, shrunk by `inset`, is projected back through the same
    camera. That box is a genuine rectangle in the film plane -- unlike the
    frame border, which is not -- so it carries the trapezoid a real rebate
    would, and no more.
    """
    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float)
    world = []
    for p in corners:
        d = R @ (np.linalg.inv(K) @ np.array([p[0], p[1], 1.0]))
        world.append(d * (25.0 / d[2]))
    world = np.array(world)
    lo, hi = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
    c = 0.5 * (lo + hi)
    lo = c + (lo - c) * (1.0 - inset)
    hi = c + (hi - c) * (1.0 - inset)
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    img = []
    for xy in box:
        q = K @ (R.T @ np.array([xy[0], xy[1], 25.0]))
        img.append(q[:2] / q[2])
    img = np.array(img)
    segs = np.array([[img[0], img[1]], [img[1], img[2]], [img[2], img[3]], [img[3], img[0]]])
    return segs, np.array(["hx", "v", "hx", "v"])


def run(pitch_deg, yaw_deg):
    print(f"\n=== pitch {pitch_deg} / yaw {yaw_deg}, f={F:.0f}px, {W}x{H} ===")
    R = rot(np.radians(-pitch_deg), np.radians(yaw_deg))
    segs = project(world_segments(), R)
    P = np.array([(s[0], s[1]) for s in segs], float)
    fam = np.array([s[2] for s in segs])

    def show(tag, Q):
        v, h, c = metrics(Q, fam)
        print(f"  {tag:<38} vert {v:7.3f}  horiz {h:7.3f}  vconv {c:6.2f}")

    show("original", P)
    show("exact rectification (= Guided)", apply_H(K @ R @ np.linalg.inv(K), P))
    Rd = rot(np.radians(-pitch_deg) * 0.953, np.radians(yaw_deg) * 0.932)
    show("Lightroom Full (doc's removal)", apply_H(K @ Rd @ np.linalg.inv(K), P))

    # What NegPy actually fits: the gate edge, not the picture. The gate is a
    # rectangle in the film plane, so its *image* is the true quad; the 2-DOF
    # family can only approximate that quad. Fit on the gate lines.
    G, gf = gate_quad(R)
    ag, cvg, chg = fit_negpy(G, gf)
    Gout = negpy_map(G, ag, cvg, chg)
    gres = np.abs(tilt(Gout, gf))
    print(f"  true gate quad corners (px): {np.round(G[:,0],1).tolist()} {np.round(G[:,1],1).tolist()}")
    print(f"  NegPy gate fit: angle {ag:+.3f} cv {cvg:+.3f}% ch {chg:+.3f}%  "
          f"(gate residual vert {np.mean(gres[gf=='v']):.3f} horiz {np.mean(gres[gf=='hx']):.3f} deg)")
    show("NegPy gate-fitted (what the solver does)", negpy_map(P, ag, cvg, chg))

    ag2, cvg2, chg2 = fit_negpy(G, gf, cap=5.0)
    print(f"  NegPy gate fit, capped: angle {ag2:+.3f} cv {cvg2:+.3f}% ch {chg2:+.3f}%")
    show("NegPy gate-fitted, capped at 5", negpy_map(P, ag2, cvg2, chg2))

    # structural quad error: how far the family's best quad sits from the truth
    fam_q = negpy_map(G, ag, cvg, chg)
    err = np.max(np.abs(fam_q - G))
    print(f"  best family quad vs true gate quad: max corner error {err:.1f} px ({100*err/W:.2f}% of width)")


if __name__ == "__main__":
    for p in (22, 15, 10, 5, 30, 40):
        run(p, 6)
