"""
Reference implementation of the perceptual "upright adjustment" energy from:

    Hyunjoon Lee, Eli Shechtman, Jue Wang, Seungyong Lee,
    "Automatic Upright Adjustment of Photographs", CVPR 2012.
    DOI 10.1109/CVPR.2012.6247761

This is the *distinctive* part of Adobe Lightroom's Upright feature: not the
vanishing-point detection (textbook), but the energy that decides HOW FAR to
correct. It deliberately does not fully rectify.

Pipeline:  detect_lines -> estimate_vanishing_points -> upright_transform
The first two are deliberately simple/classical; the third is the paper.

Pure numpy/scipy/Pillow. No OpenCV.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import minimize

# Paper's adaptive-weight schedule (section 3.2)
SIGMA_V = np.pi / 12.0
SIGMA_H = np.pi / 15.0

E_X = np.array([1.0, 0.0])
E_Y = np.array([0.0, 1.0])
_EPS = 1e-12


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def _safe_2d(v: np.ndarray) -> np.ndarray:
    """Homogeneous -> Euclidean, tolerating points at infinity."""
    if abs(v[2]) > 1e-9:
        return v[:2] / v[2]
    d = _normalize(v[:2])
    return d * 1e6


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx_, sx = np.cos(rx), np.sin(rx)
    cy_, sy = np.cos(ry), np.sin(ry)
    cz_, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rz = np.array([[cz_, -sz, 0], [sz, cz_, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def homography_from_params(x: np.ndarray, K0: np.ndarray) -> np.ndarray:
    """x = [fx, fy, rx, ry, rz, tx, ty].  H = K1 R^T K0^-1 + translation."""
    fx, fy, rx, ry, rz, tx, ty = x
    K1 = intrinsics(fx, fy, K0[0, 2], K0[1, 2])
    H = K1 @ rotation(rx, ry, rz).T @ np.linalg.inv(K0)
    H[0, 2] += tx
    H[1, 2] += ty
    return H


def _apply(H: np.ndarray, homogeneous_points: np.ndarray) -> np.ndarray:
    return (H @ homogeneous_points.T).T


def _line_through(a_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    return _cross(a_h, b_h)


def _line_direction(l: np.ndarray) -> np.ndarray:
    return _normalize(np.array([-l[1], l[0]]))


# --------------------------------------------------------------------------
# line detection (LSD-lite: Sobel + NMS + Hough + per-line TLS refit)
# --------------------------------------------------------------------------
def detect_lines(gray: np.ndarray, max_side: int = 1200, min_len: int = 40):
    """Return (N,4) array of segments x1,y1,x2,y2 in *original* image coords."""
    h0, w0 = gray.shape
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        from PIL import Image

        img = Image.fromarray(gray).resize(
            (int(w0 * scale), int(h0 * scale)), Image.BILINEAR
        )
        g = np.asarray(img, dtype=np.float64)
    else:
        g = gray.astype(np.float64)

    g = gaussian_filter(g, 1.2)
    gx, gy = sobel(g, axis=1), sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag, 88.0)
    strong = mag > max(thresh, 1e-6)

    # non-maximum suppression across the gradient direction
    ang = np.arctan2(gy, gx)
    dirs = np.round(ang / (np.pi / 4.0)).astype(int) % 4
    nms = np.zeros_like(mag)
    for d, (dy, dx) in enumerate([(0, 1), (-1, 1), (-1, 0), (-1, -1)]):
        nms[dirs == d] = mag[dirs == d]
    shifted = np.roll(np.roll(mag, 1, axis=0), 1, axis=1)
    shifted2 = np.roll(np.roll(mag, -1, axis=0), -1, axis=1)
    edges = strong & (mag >= shifted) & (mag >= shifted2)
    ys, xs = np.nonzero(edges)
    if len(xs) < 50:
        return np.zeros((0, 4))

    pts = np.stack([xs, ys], axis=1).astype(np.float64)

    # Hough over theta only, refit each peak with total-least-squares
    thetas = np.deg2rad(np.arange(0.0, 180.0, 0.5))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    rho = pts[:, 0:1] * cos_t[None, :] + pts[:, 1:2] * sin_t[None, :]
    rmin = rho.min()
    nrho = int(np.ceil((rho.max() - rmin) / 2.0)) + 1
    acc = np.zeros((len(thetas), nrho), dtype=np.int32)
    ri = np.clip(((rho - rmin) / 2.0).astype(np.int32), 0, nrho - 1)
    for t in range(len(thetas)):
        acc[t] = np.bincount(ri[:, t], minlength=nrho)
    acc = gaussian_filter(acc.astype(np.float64), 1.0)

    segments = []
    used = np.zeros(len(pts), dtype=bool)
    flat_order = np.argsort(acc, axis=None)[::-1]
    for idx in flat_order[:400]:
        t_i, r_i = np.unravel_index(idx, acc.shape)
        if acc[t_i, r_i] < min_len * scale * 0.5:
            break
        theta = thetas[t_i]
        r = r_i * 2.0 + rmin
        n = np.array([np.cos(theta), np.sin(theta)])
        d = np.array([-np.sin(theta), np.cos(theta)])
        dist = np.abs((pts - n * r) @ n)
        band = (dist < 1.5) & (~used)
        if band.sum() < min_len * scale:
            continue
        bp = pts[band]
        proj = bp @ d
        order = np.argsort(proj)
        bp, proj = bp[order], proj[order]
        # split into contiguous runs
        gaps = np.nonzero(np.diff(proj) > 6.0)[0]
        for run in np.split(np.arange(len(bp)), gaps + 1):
            if len(run) < min_len * scale:
                continue
            P = bp[run]
            c = P.mean(axis=0)
            u, s, vt = np.linalg.svd(P - c, full_matrices=False)
            if s[0] < _EPS:
                continue
            direction = vt[0]
            t_proj = (P - c) @ direction
            p1 = c + direction * t_proj.min()
            p2 = c + direction * t_proj.max()
            length = float(t_proj.max() - t_proj.min())
            if length < min_len * scale:
                continue
            segments.append([p1[0], p1[1], p2[0], p2[1], length])
            used |= band

    if not segments:
        return np.zeros((0, 4))
    seg = np.array(segments)
    if scale < 1.0:
        seg[:, :4] /= scale
    return seg


# --------------------------------------------------------------------------
# vanishing point estimation (RANSAC over line pairs, greedy decimation)
# --------------------------------------------------------------------------
def line_homogeneous(seg: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
    return np.cross(np.array([x1, y1, 1.0]), np.array([x2, y2, 1.0]))


def _angle_residual(lines_n, mids, v) -> np.ndarray:
    """Angle (rad) between each line and the ray from its midpoint to VP v.

    Zero iff the VP lies on the line. Works for VPs at infinity.
    """
    dirs = np.stack([-lines_n[:, 1], lines_n[:, 0]], axis=1)  # line directions
    if abs(v[2]) > 1e-9:
        vp2 = v[:2] / v[2]
    else:
        vp2 = _normalize(v[:2]) * 1e7
    u = vp2[None, :] - mids
    n = np.linalg.norm(u, axis=1, keepdims=True)
    n[n < 1e-9] = 1e-9
    u = u / n
    cos = np.abs(np.sum(dirs * u, axis=1))
    return np.arccos(np.clip(cos, 0.0, 1.0))


def estimate_vanishing_points(segments: np.ndarray, w: int, h: int, n_families: int = 3,
                              max_pairs: int = 4000, tol_deg: float = 2.0, seed: int = 0):
    """Greedy RANSAC: repeatedly find the highest-support VP, remove its inliers."""
    if len(segments) < 3:
        return []
    rng = np.random.default_rng(seed)
    lines = np.array([line_homogeneous(s) for s in segments])
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + _EPS
    lengths = segments[:, 4]
    mid = np.stack([(segments[:, 0] + segments[:, 2]) / 2.0,
                    (segments[:, 1] + segments[:, 3]) / 2.0], axis=1)
    live = np.ones(len(segments), dtype=bool)
    tol = np.deg2rad(tol_deg)
    vps = []

    for _ in range(n_families):
        idx = np.nonzero(live)[0]
        if len(idx) < 3:
            break
        n_pairs = min(max_pairs, len(idx) * (len(idx) - 1) // 2)
        a = rng.choice(idx, size=n_pairs)
        b = rng.choice(idx, size=n_pairs)
        ok = a != b
        a, b = a[ok], b[ok]
        cands = np.cross(lines[a], lines[b])
        norms = np.linalg.norm(cands[:, :2], axis=1)
        cands = cands[norms > 1e-9]
        if len(cands) == 0:
            break
        if len(cands) > 2500:
            cands = cands[rng.choice(len(cands), 2500, replace=False)]

        best, best_score = None, -1.0
        for v in cands:
            vn = v / (np.linalg.norm(v) + _EPS)
            inl = _angle_residual(lines[idx], mid[idx], vn) < tol
            score = float(lengths[idx][inl].sum())
            if score > best_score:
                best_score, best = score, vn
        if best is None:
            break

        # refit: VP = smallest singular vector of inlier line matrix
        inl_idx = idx[_angle_residual(lines[idx], mid[idx], best) < tol]
        if len(inl_idx) < 3:
            break
        _, _, vt = np.linalg.svd(lines[inl_idx])
        vn = vt[-1]
        if vn[2] < 0:
            vn = -vn
        inl_idx = idx[_angle_residual(lines[idx], mid[idx], vn) < tol]
        if len(inl_idx) < 3:
            break
        vps.append({"vp": vn, "inliers": inl_idx,
                    "weight": float(lengths[inl_idx].sum())})
        live[inl_idx] = False

    return vps


def classify_families(vps, segments):
    """Label families: 'vertical' = image-mostly-vertical line family."""
    out = {"vertical": None, "horizontals": []}
    for v in vps:
        dirs = []
        for i in v["inliers"]:
            s = segments[i]
            d = _normalize(np.array([s[2] - s[0], s[3] - s[1]]))
            dirs.append(abs(d[1]))
        if dirs and np.mean(dirs) > 0.5 and out["vertical"] is None:
            out["vertical"] = v
        else:
            out["horizontals"].append(v)
    return out


# --------------------------------------------------------------------------
# THE PAPER'S ENERGY  (section 3)
# --------------------------------------------------------------------------
def energy(x, K0, vert_lines, horiz_lines, vp_vert, vp_h_x, vp_h_z,
           f0, edge_pts=None, junctions=None,
           w_pic=1.0, w_eye=1.0, w_img=1.0, w_f=30.0, w_perkins=0.5,
           lam_v=None, lam_h=None):
    """
    E = E_eye + E_pic + E_img + E_f  (+ Perkins junction penalty)

    vert_lines / horiz_lines: (N,5) x1,y1,x2,y2,weight
    vp_*: homogeneous 3-vectors, the *source* vanishing points.
    lam_v / lam_h: adaptive picture-frame weights (computed from initial rotation).
    """
    H = homography_from_params(x, K0)
    total_w = (vert_lines[:, 4].sum() if len(vert_lines) else 0.0) + \
              (horiz_lines[:, 4].sum() if len(horiz_lines) else 0.0) + _EPS

    # --- E_eye : the horizon must come out horizontal -------------------
    e_eye = 0.0
    if vp_h_x is not None and vp_h_z is not None:
        lh = _line_through(_apply(H, vp_h_x[None, :])[0],
                           _apply(H, vp_h_z[None, :])[0])
        d = _line_direction(lh)
        e_eye = total_w * float(E_Y @ d) ** 2

    # --- E_pic : align structure to the picture frame -------------------
    e_pic = 0.0
    fxv = lam_v if lam_v is not None else 1.0
    fxh = lam_h if lam_h is not None else 1.0
    if vp_vert is not None and len(vert_lines):
        e_pic += fxv * _family_residual(H, vert_lines, vp_vert, E_X)
    if vp_h_x is not None and len(horiz_lines):
        e_pic += fxh * _family_residual(H, horiz_lines, vp_h_x, E_Y)

    # --- E_img : Jacobian anisotropy on curved features -----------------
    e_img = 0.0
    if edge_pts is not None and len(edge_pts) and w_img > 0:
        p = np.concatenate([edge_pts, np.ones((len(edge_pts), 1))], axis=1)
        q = _apply(H, p)
        W = q[:, 2:3]
        W = np.where(np.abs(W) < 1e-9, 1e-9, W)
        X = q[:, 0:1] / W
        Y = q[:, 1:2] / W
        J = np.empty((len(p), 2, 2))
        J[:, 0, 0] = (H[0, 0] - X[:, 0] * H[2, 0]) / W[:, 0]
        J[:, 0, 1] = (H[0, 1] - X[:, 0] * H[2, 1]) / W[:, 0]
        J[:, 1, 0] = (H[1, 0] - Y[:, 0] * H[2, 0]) / W[:, 0]
        J[:, 1, 1] = (H[1, 1] - Y[:, 0] * H[2, 1]) / W[:, 0]
        # singular values of 2x2 in closed form
        a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
        tr = a * a + b * b + c * c + d * d
        det = a * d - b * c
        disc = np.sqrt(np.maximum((tr * tr - 4 * det * det), 0.0))
        s1 = np.sqrt(np.maximum((tr + disc) / 2.0, 1e-12))
        s2 = np.sqrt(np.maximum((tr - disc) / 2.0, 1e-12))
        r = s1 / s2
        e_img = w_img * float(np.mean((r - 1.0 / r) ** 2))

    # --- E_f : keep apparent focal length plausible ---------------------
    e_f = w_f * ((x[0] / f0 - 1.0) ** 2 + (x[1] / f0 - 1.0) ** 2)

    # --- Perkins junction constraint ------------------------------------
    # Lines transform by H^-T (points by H); constrain the corrected corner angle.
    e_perk = 0.0
    if junctions is not None and len(junctions) and w_perkins > 0:
        lv = np.array([j[0] for j in junctions])
        lh = np.array([j[1] for j in junctions])
        HiT = np.linalg.inv(H).T
        At = (HiT @ lv.T).T
        Bt = (HiT @ lh.T).T
        dv = _normalize_rows(np.stack([-At[:, 1], At[:, 0]], axis=1))
        dh = _normalize_rows(np.stack([-Bt[:, 1], Bt[:, 0]], axis=1))
        ang = np.arccos(np.clip(np.abs(np.sum(dv * dh, axis=1)), 0.0, 1.0))
        e_perk = w_perkins * float(np.sum(np.maximum(0.0, np.pi / 2.0 - ang) ** 2))

    return w_eye * e_eye + w_pic * e_pic + e_img + e_f + e_perk


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < _EPS] = _EPS
    return v / n


def _family_residual(H, lines_xyw, vp, axis) -> float:
    """Sum w_i * (axis . image_line_direction)^2, fully vectorised."""
    mids = np.stack([(lines_xyw[:, 0] + lines_xyw[:, 2]) / 2.0,
                     (lines_xyw[:, 1] + lines_xyw[:, 3]) / 2.0,
                     np.ones(len(lines_xyw))], axis=1)
    q = _apply(H, mids)                      # (N,3) transformed midpoints
    vp_t = _apply(H, vp[None, :])[0]         # (3,)
    l = np.cross(q, vp_t)                    # (N,3) image lines through VP
    d = _normalize_rows(np.stack([-l[:, 1], l[:, 0]], axis=1))
    res = d @ axis
    return float(np.sum(lines_xyw[:, 4] * res ** 2))


def initial_lambdas(rx: float, ry: float):
    """Paper's adaptive schedule: picture-frame alignment relaxes as tilt grows."""
    lam_v = float(np.exp(-(rx ** 2) / (2.0 * SIGMA_V ** 2)))
    lam_h = float(np.exp(-(ry ** 2) / (2.0 * SIGMA_H ** 2)))
    return lam_v, lam_h


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def upright_transform(gray, f0=None, mode="auto", max_side=1200, verbose=False):
    """
    mode: 'level' | 'vertical' | 'auto' | 'full'
    Returns dict(H=3x3 correction, params, vps, segments)
    """
    h, w = gray.shape
    if f0 is None:
        f0 = 1.2 * max(w, h)
    K0 = intrinsics(f0, f0, w / 2.0, h / 2.0)

    segments = detect_lines(gray, max_side=max_side)
    if len(segments) < 4:
        return {"H": np.eye(3), "params": None, "vps": [], "segments": segments}

    vps = estimate_vanishing_points(segments, w, h)
    fam = classify_families(vps, segments)

    vert = fam["vertical"]
    horiz = fam["horizontals"]
    if vert is None or not horiz:
        if verbose:
            print("insufficient line structure")
        return {"H": np.eye(3), "params": None, "vps": vps, "segments": segments}

    vp_vert = vert["vp"]
    vp_h_x = horiz[0]["vp"]
    vp_h_z = horiz[1]["vp"] if len(horiz) > 1 else None

    vert_lines = segments[vert["inliers"]]
    horiz_lines = segments[horiz[0]["inliers"]]

    # initial rotation from VP geometry -> sets the adaptive lambdas
    K0i = np.linalg.inv(K0)
    d_v = _normalize(K0i @ vp_vert)
    if d_v[2] < 0:
        d_v = -d_v
    pitch0 = float(np.arctan2(-d_v[2], abs(d_v[1]) + _EPS))
    d_h = _normalize(K0i @ vp_h_x)
    if d_h[2] < 0:
        d_h = -d_h
    yaw0 = float(np.arctan2(d_h[0], abs(d_h[2]) + _EPS))

    lam_v, lam_h = initial_lambdas(pitch0, yaw0)
    weights = {
        "level":    dict(w_pic=1.0, w_eye=4.0, w_img=1.0, w_f=30.0),
        "vertical": dict(w_pic=1.0, w_eye=1.0, w_img=1.0, w_f=30.0),
        "auto":     dict(w_pic=1.0, w_eye=1.0, w_img=1.0, w_f=30.0),
        "full":     dict(w_pic=6.0, w_eye=4.0, w_img=0.3, w_f=10.0),
    }
    if mode == "level":
        lam_v, lam_h = 0.15, 3.0
    elif mode == "vertical":
        lam_v, lam_h = 3.0, 0.15
    elif mode == "full":
        lam_v = lam_h = 8.0

    edge_pts = _edge_points(gray, max_side)
    junctions = _junctions(vert_lines, horiz_lines, w, h)

    x0 = np.array([f0, f0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = minimize(
        energy, x0, method="Powell",
        args=(K0, vert_lines, horiz_lines, vp_vert, vp_h_x, vp_h_z, f0,
              edge_pts, junctions),
        options=dict(**weights[mode], lam_v=lam_v, lam_h=lam_h,
                     maxiter=4000, xtol=1e-6, ftol=1e-9),
    )
    H = homography_from_params(res.x, K0)
    return {"H": H, "params": res.x, "vps": vps, "segments": segments,
            "mode": mode, "f0": f0, "lam": (lam_v, lam_h)}


def _edge_points(gray, max_side, n_max=1500):
    from PIL import Image
    h, w = gray.shape
    s = min(1.0, max_side / max(h, w))
    g = np.asarray(Image.fromarray(gray).resize((int(w * s), int(h * s))), dtype=np.float64)
    g = gaussian_filter(g, 1.0)
    gx, gy = sobel(g, axis=1), sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    m = mag > np.percentile(mag, 92.0)
    ys, xs = np.nonzero(m)
    if len(xs) > n_max:
        idx = np.random.default_rng(1).choice(len(xs), n_max, replace=False)
        xs, ys = xs[idx], ys[idx]
    return np.stack([xs / s, ys / s], axis=1)


def _junctions(vert_lines, horiz_lines, w, h, max_n=200):
    out = []
    for lv in vert_lines[:max_n]:
        for lh in horiz_lines[:max_n]:
            a = np.cross(np.array([lv[0], lv[1], 1.0]), np.array([lv[2], lv[3], 1.0]))
            b = np.cross(np.array([lh[0], lh[1], 1.0]), np.array([lh[2], lh[3], 1.0]))
            p = np.cross(a, b)
            if abs(p[2]) < 1e-9:
                continue
            px, py = p[0] / p[2], p[1] / p[2]
            if 0 <= px < w and 0 <= py < h:
                out.append((a, b))
    return out


# --------------------------------------------------------------------------
# warp + crop
# --------------------------------------------------------------------------
def warp(img: np.ndarray, H: np.ndarray, out_shape=None):
    h, w = img.shape[:2]
    if out_shape is None:
        out_shape = (h, w)
    oh, ow = out_shape
    yy, xx = np.mgrid[0:oh, 0:ow]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(oh * ow)], axis=0)
    Hi = np.linalg.inv(H)
    q = Hi @ p
    W = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    sx, sy = q[0] / W, q[1] / W
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = sx - x0, sy - y0
    src = img.astype(np.float64)
    if src.ndim == 2:
        src = src[:, :, None]
    out = np.empty((oh * ow, src.shape[2]))
    for c in range(src.shape[2]):
        top = src[y0, x0, c] * (1 - fx) + src[y0, x1, c] * fx
        bot = src[y1, x0, c] * (1 - fx) + src[y1, x1, c] * fx
        out[:, c] = top * (1 - fy) + bot * fy
    out[~valid] = 0.0
    out = out.reshape(oh, ow, src.shape[2])
    return out[:, :, 0] if img.ndim == 2 else out


def constrain_crop(H, w, h, steps=60):
    """Largest axis-aligned box (centred on the warped quad's centroid) that fits
    entirely inside the warped source quad.

    Returns (scale, centre_in_output_px). The caller should then apply
        S(scale about centre) @ H
    and render at the original canvas size with the centre moved to w/2, h/2.
    """
    quad = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    q = (H @ quad.T).T
    q = q[:, :2] / q[:, 2:3]
    c = q.mean(axis=0)

    def fits(s):
        box = np.array([[c[0] - w * s / 2, c[1] - h * s / 2],
                        [c[0] + w * s / 2, c[1] - h * s / 2],
                        [c[0] + w * s / 2, c[1] + h * s / 2],
                        [c[0] - w * s / 2, c[1] + h * s / 2]])
        return all(_point_in_quad(b, q) for b in box)

    lo, hi = 0.0, 1.0
    if not fits(1.0):
        for _ in range(steps):
            mid = (lo + hi) / 2
            if fits(mid):
                lo = mid
            else:
                hi = mid
    else:
        lo = 1.0
    return lo, c


def crop_matrix_of(scale, centre, w, h):
    """Map output-space point p -> scale*(p - centre) + (w/2, h/2)."""
    T1 = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]], dtype=float)
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=float)
    T2 = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], dtype=float)
    return T2 @ S @ T1


def _point_in_quad(p, quad):
    sign = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) < 1e-9:
            continue
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True
