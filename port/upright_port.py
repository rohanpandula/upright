"""
upright_port -- a clean-room, licence-clean reference implementation of the
geometry behind Lightroom-style "Upright" perspective correction.

The model is the one recovered by measurement, not the textbook one::

    H = s * K1 * R * K0**-1          with K1 != K0

`K0` is the *source* camera (centered principal point, focal initialized from
metadata and checked against observed perpendicular families), `K1` is the
*destination* camera the solver invents so the output
looks natural, `R` is the rectifying rotation and `s` an irrelevant overall
scale.  Collapsing this to the single-camera form ``K * R * K**-1`` is the most
common porting mistake: measured against 29 real Adobe solves the two-camera
model reaches a median orthogonality residual of 5.7e-16 while the single-camera
form stalls at 3.2e-01 (see evidence/orthogonality.txt and model_exactness.txt).
Both cameras therefore stay distinct in this module.

Provenance
----------
Constants are tagged where they are defined:

``MEASURED``
    Fitted from the shipped Lightroom solver (CameraRaw 15.5) over controlled
    images with exactly known camera rotation plus a corpus of real solves.
    Reproducible, but an inference through an assumed response curve.
``PAPER``
    Published in, or a direct consequence of, Lee/Shechtman/Wang/Lee,
    "Automatic Upright Adjustment of Photographs", CVPR 2012,
    DOI 10.1109/CVPR.2012.6247761.
``PROVEN``
    Read out of the shipped binary as an immediate/constant (address given in
    evidence/). Not copyrightable expression, only a number.
``CHOICE``
    Our implementation policy; not established as an Adobe constant or algorithm.

Nothing here is Adobe source code: the algebra, the optimiser and the detector
are re-derived. Recovered numbers and our implementation choices are distinguished.

Public API
----------
``solve(segments, w, h, focal_px, mode)`` -> 3x3 homography in *normalized*
coordinates (x in [0, 1] across the width, y in [0, 1] down the height).
``segments`` is an (N, 4) array of ``x1, y1, x2, y2`` pixel segments.  Pass
``segments=None`` together with ``image=`` to run the bundled detector; the
detector is deliberately separable so a caller can substitute its own.

Modes: ``off``, ``level``, ``vertical``, ``full``, ``auto``, ``guided``.
``guided`` takes exactly four segments as *hard constraints* and drives them
onto the axes to numerical zero; it needs no energy model and is the mode to
trust most.

Only numpy and OpenCV are used.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

__all__ = [
    "solve",
    "solve_ex",
    "detect_segments",
    "focal_px_from_35mm",
    "focal_35mm_from_px",
    "metadata_focal_px",
    "MODES",
]

# ---------------------------------------------------------------------------
# Constants.  Every number is tagged; see the module docstring for the legend.
# ---------------------------------------------------------------------------

# Full-frame sensor: 36 x 24 mm, used as its *diagonal*.  PROVEN (binary
# constants 36.0 and 24.0, evidence/focal_prior.txt section 4).  The diagonal is
# 43.266615305567875 mm, so a 35 mm-equivalent focal maps to
# f_px = f35 * hypot(w, h) / 43.266615... -- NOT f35 / 36 * max(w, h), which is
# the usual (wrong) short-edge conversion.
SENSOR_DIAGONAL_MM = 43.266615305567875  # PROVEN

# Fallback focal when no metadata exists.  MEASURED/proven: 35.0 is the
# shipped default in four independent places including the estimation-failure
# path (evidence/focal_prior.txt section 1).
DEFAULT_FOCAL_MM = 35.0  # PROVEN

# Plausible range for a *metadata* focal, applied before conversion.  The
# geometric estimate is NOT ceilinged by this (evidence/focal_prior.txt sec. 2).
FOCAL_CLAMP_MM = (10.0, 1000.0)  # PROVEN

# Auto's vertical response is a fraction of removed visible convergence, not
# Euler pitch. MEASURED pooled fit: evidence/sigma_v_precise.txt, fit C;
# Round 4 uses the requested L0=2.64 (fit C itself reports 2.474).
SIGMA_V = math.radians(13.33)  # MEASURED, keystone units, CI 9.94 .. 22.72 deg
SIGMA_V_CI_DEG = (9.94, 22.72)  # MEASURED
L0_V = 2.64  # MEASURED/requested; pooled fit, not the old branch-conditioned 3.56
# Round 13: the shipped Auto response is a function of UNDAMPED CAMERA PITCH, not of
# the detected keystone spread. 12,075 native Adobe transforms were decomposed through
# the two-camera model and the removed convergence fraction binned by recovered input
# pitch; the curve is a lognormal hump (peak ~0.80 at 6-12 deg, ~0.17 at 30 deg, ~0 at
# 40 deg). MEASURED, n = 4,966 cross-checked photos, evidence/auto_corpus.txt.
AUTO_PITCH_KNOTS_DEG = (1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.5, 16.5, 20.0,
                        24.5, 31.0, 40.5)
AUTO_PITCH_KNOTS_FRACTION = (0.532, 0.631, 0.741, 0.815, 0.789, 0.790, 0.692,
                             0.589, 0.561, 0.468, 0.170, 0.0)
# The corpus curve is measured against full-frame geometric convergence; the four
# cached Adobe scenes sit slightly above the corpus median at their pitch, and the
# scalar 1.20 was calibrated so scene A reproduces Adobe's own 0.652. MEASURED/CAL.
AUTO_PITCH_GAIN = 1.20

# Auto's horizontal (yaw) damping.  Gated on pitch being present: on a pure-yaw
# frame Auto is a no-op, because the frame's verticals are already parallel.
LAMBDA0_H = 76.73  # MEASURED
SIGMA_YAW = math.radians(11.64)  # MEASURED/requested Round 4 value
SIGMA_PITCH_GATE = math.radians(22.10)  # MEASURED

# Roll is corrected almost unconditionally, but not quite: the shipped Auto
# applies a median 0.928 of the roll correction, and Level 0.87. MEASURED
# (evidence/lambda_curve.txt, n = 5854).
ROLL_AUTO = 0.928  # MEASURED
ROLL_LEVEL = 0.87  # MEASURED

# Paper's adaptive-weight sigmas, kept for A/B testing only.  PAPER.
SIGMA_V_PAPER = math.pi / 12.0  # PAPER (DOI 10.1109/CVPR.2012.6247761)
SIGMA_H_PAPER = math.pi / 15.0  # PAPER (DOI 10.1109/CVPR.2012.6247761)

# Detector constants: evidence/detector.txt sections 3-4; corrections from
# raw scalar-width/instruction audit in evidence/detector_corrections.txt.
COS_22_5 = 0.9238795042037964  # PROVEN: cos(22.5 deg), 8-direction quantisation
SIGNIFICANCE_LOG_SCALE = 4.0  # PROVEN factor: evidence/detector.txt section 3.1
SIGNIFICANCE_BASE = 8.0  # PROVEN: log base 8, evidence/detector.txt section 3.1
RANSAC_ITERS = 2000  # PROVEN value: evidence/vp_resolution.txt; port approximation uses iterations
MIN_INLIERS = 3  # PROVEN: evidence/vp_resolution.txt
TAU = math.sqrt(3.0)  # PROVEN clamp: evidence/vp_resolution.txt; port uses an approximate gate
MAX_VP_CANDIDATES = 9  # PROVEN: evidence/detector.txt section 4.1
MERGE_RADIUS_PX = 6  # PROVEN: 0x22d2b60(obj, 6, 10.0f)
MERGE_PERP_PX = float(MERGE_RADIUS_PX)  # PROVEN: detector_corrections.txt, NOT 12
MERGE_GAP_PX = 10.0  # PROVEN: strict interval-containment slack, corrections.txt
DIRECTION_COUNT = 8  # PROVEN: detector.txt section 3.2
DIRECTIONS = np.arange(DIRECTION_COUNT) * (math.pi / DIRECTION_COUNT)  # PROVEN spec: detector.txt 3.2
NFA_P = 0.125  # PROVEN: detector+0x34, detector_corrections.txt (NOT interpolation)
NFA_TEST_EXPONENT = 4  # PROVEN: 0x22cb425..0x22cb42c, detector_corrections.txt
REFINE_GAP_PX = 10  # PROVEN: esi at 0x22d22a6, detector_corrections.txt
REFINE_DISTANCE_PX = 1.0  # PROVEN: movss at 0x22d22ae, detector_corrections.txt
REFINE_PASSES = 99  # PROVEN: detector.txt section 3.1
ANGULAR_TOLERANCES_SQ = (1 / 3600, 1 / 8100, 1 / 900)  # PROVEN: detector.txt 4.2
CAMERA_TOLERANCE_SQ = 0.16  # PROVEN value; role UNRESOLVED, detector.txt 4.2

# Working-image preprocessing choices, not recovered Adobe constants.
DETECTOR_BLUR_SIGMA = 1.2  # CHOICE: evidence/accuracy_gap.txt
DETECTOR_EDGE_PERCENTILE = 85.0  # CHOICE: evidence/accuracy_gap.txt

DEFAULT_MAX_SIDE = 1200  # CHOICE: working policy, evidence/accuracy_gap.txt

MODES = ("off", "level", "vertical", "full", "auto", "guided")

# Shared full-solve weights. PAPER term structure / CHOICE weights,
# evidence/accuracy_gap.txt; NOT Adobe weights.
_FULL_WEIGHTS = dict(w_pic=6.0, w_eye=4.0, w_f=10.0)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Focal-length conversion
# ---------------------------------------------------------------------------
def focal_px_from_35mm(f35: float, w: int, h: int) -> float:
    """Convert a 35 mm-equivalent focal to pixels via the full-frame diagonal.

    ``f_px = f35 * hypot(w, h) / 43.266615305567875``

    A 36 x 24 mm frame gives back exactly ``f35``.  No clamping is applied here,
    so a geometric estimate may legitimately exceed the metadata range.
    """
    return float(f35) * math.hypot(float(w), float(h)) / SENSOR_DIAGONAL_MM


def focal_35mm_from_px(focal_px: float, w: int, h: int) -> float:
    """Inverse of :func:`focal_px_from_35mm`."""
    return float(focal_px) * SENSOR_DIAGONAL_MM / math.hypot(float(w), float(h))


def metadata_focal_px(f35: float | None, w: int, h: int) -> float:
    """Convert an EXIF focal to pixels, clamping the *metadata* value first.

    Missing or non-finite metadata falls back to 35 mm.  The clamp to
    [10, 1000] mm applies only here; a geometric estimate must be passed
    straight to :func:`focal_px_from_35mm` instead.
    """
    if f35 is None or not np.isfinite(f35) or f35 <= 0.0:
        f35 = DEFAULT_FOCAL_MM
    lo, hi = FOCAL_CLAMP_MM
    f35 = min(hi, max(lo, float(f35)))
    return focal_px_from_35mm(f35, w, h)


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < _EPS] = _EPS
    return v / n


def _intrinsics_norm(focal_px: float, w: int, h: int) -> np.ndarray:
    """Normalized-coordinate camera matrix (centre 0.5, 0.5)."""
    return np.array(
        [[focal_px / w, 0.0, 0.5], [0.0, focal_px / h, 0.5], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def _polar(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def _euler_xyz(R: np.ndarray) -> np.ndarray:
    """Decompose R = Rz Ry Rx into (rx, ry, rz)."""
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(math.cos(ry)) > 1e-8:
        rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
        rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:  # gimbal lock
        rx = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        rz = 0.0
    return np.array([rx, ry, rz])


def _apply_h(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (H @ np.asarray(p, dtype=float).T).T


def _homogeneous_lines(seg_n: np.ndarray) -> np.ndarray:
    a = np.column_stack([seg_n[:, 0], seg_n[:, 1], np.ones(len(seg_n))])
    b = np.column_stack([seg_n[:, 2], seg_n[:, 3], np.ones(len(seg_n))])
    return np.cross(a, b)


# ---------------------------------------------------------------------------
# Segment detector
# ---------------------------------------------------------------------------
def _significance_bins(maxdim: int) -> int:
    """Smoothed-pass significance bins; raw fallback uses NFA directly.

    Retain the requested round() formula and existing API/tests. The binary
    actually floors (ROUNDSS immediate 9); detector_corrections.txt records it.
    """
    value = np.float32(SIGNIFICANCE_LOG_SCALE) * np.log(np.float32(max(maxdim, 1)))
    value /= np.log(np.float32(SIGNIFICANCE_BASE))
    return int(round(float(value)))


def _log_nfa(n: int, k: int, maxdim: int) -> float:
    """log(M**4 * P[Binomial(n, 1/8) >= k]), without probability underflow.

    PROVEN null and test count: evidence/detector_corrections.txt. Summing the
    exact tail replaces Adobe's overflow-prone integer binomial coefficients.
    """
    if not 0 <= k <= n or maxdim < 1:
        raise ValueError("NFA requires 0 <= k <= n and maxdim >= 1")
    log_tests = NFA_TEST_EXPONENT * math.log(maxdim)
    if k == 0:
        return log_tests
    log_p, log_q = math.log(NFA_P), math.log1p(-NFA_P)
    term = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * log_p + (n - k) * log_q)
    tail = term
    for j in range(k, n):
        term += math.log(n - j) - math.log(j + 1) + log_p - log_q
        tail = float(np.logaddexp(tail, term))
    return log_tests + min(0.0, tail)


def _nms(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Keep pixels that are maximal along the gradient direction (4-way)."""
    h, w = mag.shape
    # 0..3 quantised gradient direction
    q = (np.round(ang / (np.pi / 4.0)).astype(np.int32) % 4)
    keep = np.zeros_like(mag, dtype=bool)
    offsets = {
        0: ((0, 1), (0, -1)),      # horizontal gradient
        1: ((1, 1), (-1, -1)),     # diagonal
        2: ((1, 0), (-1, 0)),      # vertical gradient
        3: ((1, -1), (-1, 1)),     # anti-diagonal
    }
    for d, (fwd, bwd) in offsets.items():
        m = q == d
        if not m.any():
            continue
        f = np.zeros_like(mag)
        b = np.zeros_like(mag)
        fy, fx = fwd
        by, bx = bwd
        f[max(0, fy):h + min(0, fy), max(0, fx):w + min(0, fx)] = \
            mag[max(0, -fy):h - max(0, fy), max(0, -fx):w - max(0, fx)]
        b[max(0, by):h + min(0, by), max(0, bx):w + min(0, bx)] = \
            mag[max(0, -by):h - max(0, by), max(0, -bx):w - max(0, bx)]
        keep |= m & (mag >= f) & (mag >= b)
    return keep


def _chain_points(mask: np.ndarray) -> list[np.ndarray]:
    """Link selected edge pixels into 8-connected chains."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return []
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    order = np.argsort(lab, kind="stable")
    lab, xs, ys = lab[order], xs[order], ys[order]
    cuts = np.nonzero(np.diff(lab))[0] + 1
    chains = []
    for start, stop in zip(np.r_[0, cuts], np.r_[cuts, len(lab)]):
        if stop - start < 3:
            continue
        chains.append(np.column_stack([xs[start:stop],
                                       ys[start:stop]]).astype(float))
    return chains


def _fit_chain(points):
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    direction = vt[0]
    t = np.sum((points - centre) * direction, axis=1)
    return centre, direction, t


def _split_gaps(points: np.ndarray, gap: float) -> list[np.ndarray]:
    """Return point arrays (not indices into an accidentally unsorted chain)."""
    if len(points) < 2:
        return []
    _, _, t = _fit_chain(points)
    order = np.argsort(t, kind="stable")
    cut = np.flatnonzero(np.diff(t[order]) > gap) + 1
    return [part for part in np.split(points[order], cut) if len(part) >= 2]


def _refine_chains(chains, gap_limit=REFINE_GAP_PX, distance=REFINE_DISTANCE_PX):
    """Refit and split non-straight supports, with the recovered call arguments.

    The TLS split itself is a clean-room approximation; the notes do not specify
    Adobe's support reassignment. CHOICE: evidence/accuracy_gap.txt.
    """
    for _ in range(REFINE_PASSES):
        refined = []
        changed = False
        for points in chains:
            for run in _split_gaps(points, gap_limit):
                c, d, t = _fit_chain(run)
                perpendicular = np.abs(np.sum((run - c) * [-d[1], d[0]], axis=1))
                if float(perpendicular.max()) > distance and len(run) > 2:
                    order = np.argsort(t, kind="stable")
                    halves = np.array_split(run[order], 2)
                    refined.extend(part for part in halves if len(part) >= 2)
                    changed = True
                else:
                    refined.append(run)
        chains = refined
        if not changed:
            break
    return chains


def detect_segments(gray: np.ndarray, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    """Return pixel segments, retaining the ``solve(segments=...)`` seam.

    Recovered: eight directions, angular test, binomial NFA, refine/merge
    arguments. An unsmoothed fallback avoids erasing short edges. The smoothed
    pass retains its conservative bin gate; the fallback uses NFA alone. Each
    pass requires NFA < 1 (combined nominal budget <= 2). Preprocessing and TLS
    splitting are our choices; see evidence/port_review.txt.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.ndim != 2 or not gray.size or not np.isfinite(gray).all():
        raise ValueError("image must be a non-empty finite grayscale or BGR array")
    if max_side < 1:
        raise ValueError("max_side must be positive")
    h0, w0 = gray.shape
    scale = min(1.0, float(max_side) / max(h0, w0))
    img = gray.astype(np.float32)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                         interpolation=cv2.INTER_AREA)
    for sigma in (DETECTOR_BLUR_SIGMA, 0.0):
        seg = _detect_gradient_segments(img, sigma)
        if len(seg):
            return _merge_segments(seg, MERGE_RADIUS_PX, MERGE_GAP_PX) / scale
    return np.zeros((0, 4))


def _detect_gradient_segments(img: np.ndarray, sigma: float) -> np.ndarray:
    h, w = img.shape
    maxdim = max(h, w)
    g = cv2.GaussianBlur(img, (0, 0), sigma) if sigma else img
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if float(mag.max()) <= _EPS:
        return np.zeros((0, 4))
    threshold = max(float(np.percentile(mag, DETECTOR_EDGE_PERCENTILE)), _EPS)
    edges = _nms(mag, np.arctan2(gy, gx)) & (mag > threshold)
    ux = gx / np.maximum(mag, _EPS)
    uy = gy / np.maximum(mag, _EPS)
    # MEASURED: quadratic localization of a gradient maximum reduces the
    # raster staircase bias; evidence/accuracy_gap.txt (not a binary claim).
    yy, xx = np.indices(img.shape, dtype=np.float32)
    before = cv2.remap(mag, xx - ux, yy - uy, cv2.INTER_LINEAR)
    after = cv2.remap(mag, xx + ux, yy + uy, cv2.INTER_LINEAR)
    curvature = before - 2 * mag + after
    offset = np.zeros_like(mag)
    concave = curvature < -_EPS
    offset[concave] = np.clip((before[concave] - after[concave]) /
                              (2 * curvature[concave]), -0.5, 0.5)
    # Preserve the existing smoothed candidate policy; short raw supports need
    # no rounded-bin approximation because NFA already tests significance.
    n_bins = _significance_bins(maxdim) if sigma else 0
    segments = []
    for theta in DIRECTIONS:
        aligned = np.abs(ux * math.sin(theta) + uy * math.cos(theta)) >= COS_22_5
        chains = [c for c in _chain_points(edges & aligned) if len(c) > n_bins]
        for chain in chains:
            xy = chain.astype(int)
            y, x = xy[:, 1], xy[:, 0]
            chain[:, 0] += offset[y, x] * ux[y, x]
            chain[:, 1] += offset[y, x] * uy[y, x]
        for points in _refine_chains(chains, REFINE_GAP_PX, REFINE_DISTANCE_PX):
            if len(points) <= n_bins:
                continue
            c, d, t = _fit_chain(points)
            xy = np.rint(points).astype(int)
            # Validate against the FITTED normal, including unaligned samples.
            dot = np.abs(-d[1] * ux[xy[:, 1], xy[:, 0]]
                         + d[0] * uy[xy[:, 1], xy[:, 0]])
            k = int(np.count_nonzero(dot >= COS_22_5))
            if _log_nfa(len(points), k, maxdim) >= 0.0:
                continue
            a, b = c + d * t.min(), c + d * t.max()
            if np.linalg.norm(b - a) > _EPS:
                segments.append([*a, *b])
    return np.asarray(segments, dtype=float).reshape(-1, 4)


def _merge_segments(seg: np.ndarray, radius=MERGE_RADIUS_PX,
                    slack=MERGE_GAP_PX) -> np.ndarray:
    """Absorb near-contained supports, preserving the surviving fitted line.

    PROVEN: perpendicular <= radius, start delta > -slack, end delta < slack
    (0x22d2dac..0x22d2dfa), evidence/detector_corrections.txt. No angular gate.
    """
    seg = np.asarray(seg, dtype=float).reshape(-1, 4).copy()
    order = np.argsort(-np.linalg.norm(seg[:, 2:] - seg[:, :2], axis=1), kind="stable")
    seg = seg[order]
    alive = np.ones(len(seg), dtype=bool)
    for i in range(len(seg)):
        if not alive[i]:
            continue
        a = seg[i, :2]
        d = seg[i, 2:] - a
        length = float(np.linalg.norm(d))
        if length <= _EPS:
            alive[i] = False
            continue
        d /= length
        perp = np.array([-d[1], d[0]])
        idx = np.flatnonzero(alive & (np.arange(len(seg)) > i))
        c, e = seg[idx, :2] - a, seg[idx, 2:] - a
        tc, te = np.sum(c * d, axis=1), np.sum(e * d, axis=1)
        distance = np.maximum(np.abs(np.sum(c * perp, axis=1)),
                              np.abs(np.sum(e * perp, axis=1)))
        start, end = np.minimum(tc, te), np.maximum(tc, te)
        absorb = ((distance <= radius) & (start > -slack)
                  & (end - length < slack))
        alive[idx[absorb]] = False
    return seg[alive]


# ---------------------------------------------------------------------------
# Vanishing points
# ---------------------------------------------------------------------------
def _vp_residual(
    starts: np.ndarray, mids: np.ndarray, vp: np.ndarray, tau: float = TAU
) -> np.ndarray:
    """Distance of each segment START POINT to the line (VP, midpoint).

    Zero iff the VP lies on the segment's support; finite for VPs at infinity.
    Exactly the shipped vote kernel's magnitude term, with the ``tau`` cut.
    """
    c = np.cross(np.broadcast_to(vp, starts.shape), mids)
    denom = np.hypot(c[:, 0], c[:, 1])
    denom[denom < _EPS] = _EPS
    r = np.abs(np.sum(c * starts, axis=1)) / denom
    return np.minimum(r, tau)


def estimate_vanishing_points(
    segments: np.ndarray, w: int, h: int, seed: int = 0
) -> list[dict]:
    """Multi-model RANSAC over segment pairs, then a disjoint re-assignment.

    Coordinates stay in PIXELS: the residual is a perpendicular distance, so the
    ``tau = sqrt(3)`` px gate applies directly.  It is floored at 0.2 % of the
    long image side so the gate scales with resolution; clustering assumes
    ``f ~ max(w, h)`` until a real focal is recovered later
    (evidence/detector.txt section 6, vp_clustering.txt section 6).

    Candidates are generated from every pair and scored on all segments, not on
    a shrinking live set.  A segment can be a plausible inlier of two families
    (a forward-pointing line often passes within the gate of the vertical VP),
    so after selection each segment is assigned to its nearest accepted VP and
    the families are refitted.  Greedy removal alone loses such families.
    """
    if len(segments) < MIN_INLIERS:
        return []
    seg = np.asarray(segments, dtype=float)
    lines = _homogeneous_lines(seg)
    norms = np.linalg.norm(lines[:, :2], axis=1)
    good = norms > _EPS
    if good.sum() < MIN_INLIERS:
        return []
    lines[good] /= norms[good, None]
    mids = np.column_stack([(seg[:, 0] + seg[:, 2]) * 0.5,
                            (seg[:, 1] + seg[:, 3]) * 0.5,
                            np.ones(len(seg))])
    starts = np.column_stack([seg[:, 0], seg[:, 1], np.ones(len(seg))])
    lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    weights = lengths / max(float(lengths.mean()), _EPS)

    tau = max(TAU, 0.002 * max(int(w), int(h)))
    rng = np.random.default_rng(seed)
    n = len(seg)
    n_pairs = min(RANSAC_ITERS, n * (n - 1) // 2)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return []
    cand = np.cross(lines[a], lines[b])
    cn = np.linalg.norm(cand, axis=1)
    cand = cand[cn > _EPS]
    if len(cand) == 0:
        return []
    if len(cand) > 1200:
        cand = cand[rng.choice(len(cand), 1200, replace=False)]

    scored = []
    for v in cand:
        v = _normalize(v)
        inl = _vp_residual(starts, mids, v, tau) < tau
        count = int(inl.sum())
        if count < MIN_INLIERS:
            continue
        scored.append((float(weights[inl].sum()), count, v, inl))
    if not scored:
        return []
    scored.sort(key=lambda t: -t[0])

    accepted: list[dict] = []
    covered = np.zeros(n, dtype=bool)
    for score, count, v, inl in scored:
        if len(accepted) >= MAX_VP_CANDIDATES:
            break
        # A hypothesis must explain at least MIN_INLIERS segments that nothing
        # accepted so far explains, otherwise a lucky four-line intersection
        # that merely re-covers other families outranks a real one.
        fresh = inl & ~covered
        if np.count_nonzero(fresh) < MIN_INLIERS:
            continue
        dup = False
        for existing in accepted:
            union = np.count_nonzero(existing["mask"] | inl)
            inter = np.count_nonzero(existing["mask"] & inl)
            if union and inter / union > 0.6:
                dup = True
                break
        if dup:
            continue
        accepted.append({"vp": v, "mask": inl, "weight": score})
        covered |= fresh
    if not accepted:
        return []

    # Assign every segment to its nearest accepted VP, then refit.
    vps = np.array([v["vp"] for v in accepted])
    res = np.stack([_vp_residual(starts, mids, v, tau) for v in vps], axis=1)
    best = np.argmin(res, axis=1)
    best_r = res[np.arange(n), best]

    # A line's support can pass close to several vanishing points, so distance
    # alone mixes families (a forward VP sits inside the frame and long
    # verticals brush past it).  Keep only inliers whose own orientation agrees
    # with the family's majority orientation.
    seg_vertical = np.abs(seg[:, 3] - seg[:, 1]) >= np.abs(seg[:, 2] - seg[:, 0])

    out = []
    for k, acc in enumerate(accepted):
        inl = np.nonzero((best == k) & (best_r < TAU))[0]
        if len(inl) < MIN_INLIERS:
            continue
        is_vertical = bool(np.mean(seg_vertical[inl]) > 0.5)
        inl = inl[seg_vertical[inl] == is_vertical]
        if len(inl) < MIN_INLIERS:
            continue
        _, _, vt = np.linalg.svd(lines[inl])
        v = _normalize(vt[-1])
        if v[2] < 0:
            v = -v
        inl = np.nonzero(_vp_residual(starts, mids, v, tau) < tau)[0]
        inl = inl[seg_vertical[inl] == is_vertical]
        if len(inl) < MIN_INLIERS:
            continue
        out.append({"vp": v, "inliers": inl,
                    "weight": float(weights[inl].sum())})
    out.sort(key=lambda v: -v["weight"])
    return out


def _refine_vp(segments, w, h, vp):
    """Length-weighted Cauchy angular VP fit on the returned support set.

    CHOICE: IRLS in centered isotropic coordinates, with the energy's four-
    pixel angular scale. Raw pixel TLS overweights short angular outliers.
    """
    d = max(w, h)
    N = np.array([[1/d, 0, -w/(2*d)], [0, 1/d, -h/(2*d)], [0, 0, 1.]])
    seg = (segments - [w/2, h/2, w/2, h/2]) / d
    lines = _homogeneous_lines(seg)
    lines /= np.maximum(np.linalg.norm(lines[:, :2], axis=1)[:, None], _EPS)
    mids = (seg[:, :2] + seg[:, 2:]) / 2
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    v = _normalize(N @ vp)
    for _ in range(20):
        denom = np.maximum(np.linalg.norm(v[:2] - mids*v[2], axis=1), _EPS)
        residual = np.sum(lines*v, axis=1) / denom
        weight = np.sqrt(lengths / (denom**2 * (1 + (residual/(4/max(1, d-1)))**2)))
        _, _, vt = np.linalg.svd(lines*weight[:, None], full_matrices=False)
        fitted = _normalize(vt[-1])
        change = np.linalg.norm(np.cross(v, fitted))
        v = fitted
        if change < 1e-10:
            break
    v = _normalize(np.linalg.inv(N) @ v)
    return -v if v[2] < 0 else v


def _classify(vps: list[dict], seg_n: np.ndarray,
              w: float = 1.0, h: float = 1.0) -> tuple[dict | None, list[dict]]:
    """Split families into one vertical family and the horizontal ones.

    CHOICE: compare in pixels, not normalized coordinates. seg_n is divided by
    (w, h) independently, so a 45-degree line in pixels is not 45 degrees in
    seg_n on a non-square frame, and the bias grows with aspect. Above 20:1
    every family was misclassified; on a 10000x3 frame even an exactly
    projected scene failed.
    """
    verticals, horizontals = [], []
    for v in vps:
        s = seg_n[v["inliers"]]
        dx = np.abs(s[:, 2] - s[:, 0]) * w
        dy = np.abs(s[:, 3] - s[:, 1]) * h
        (verticals if float(np.mean(dy > dx)) > 0.5 else horizontals).append(v)
    vertical = max(verticals, key=lambda v: v["weight"]) if verticals else None
    horizontals.sort(key=lambda v: -v["weight"])
    return vertical, horizontals


# ---------------------------------------------------------------------------
# Energy model (auto / level / vertical / full)
# ---------------------------------------------------------------------------
def _family_terms(H, seg_n, axis, weight, w, h, mean_length):
    # PAPER: E_pic is a length-weighted sum over observed lines, not virtual
    # lines through a fitted VP (evidence/accuracy_gap.txt, energy derivation).
    lines = _homogeneous_lines(seg_n) @ np.linalg.inv(H)
    d = _normalize_rows(np.column_stack([-lines[:, 1] * w, lines[:, 0] * h]))
    lengths = np.hypot((seg_n[:, 2] - seg_n[:, 0]) * w,
                       (seg_n[:, 3] - seg_n[:, 1]) * h)
    residual = d @ axis
    # CHOICE: Cauchy and its scale. A recovered camera tolerance inspired the
    # four-pixel value; its use in this loss is unproven (port_review.txt).
    scale = 4.0 / max(1, max(w, h) - 1)
    residual = np.sign(residual) * scale * np.sqrt(np.log1p((residual / scale) ** 2))
    return np.sqrt(weight * lengths / max(mean_length, _EPS)) * residual


def _energy_residuals(theta, K0, K0i, vert, horiz, vp_h2, f0,
                      weights, lam_v, lam_h):
    """Residual vector: observed picture-frame lines, horizon, destination focal.

    ``vert`` is ``(segments, vp)``.  Only the *lateral* horizontal family is
    driven onto the horizontal axis; a third (depth) family is not horizontal
    after rectification -- its lines radiate from the principal point -- and is
    represented only through the horizon term, exactly as in the paper.  The
    first horizontal family here is the lateral one, ``vp_h2`` the depth one.
    """
    rx, ry, rz, log_f1, c1x, c1y = theta
    f1 = float(np.exp(log_f1))
    K1 = np.array([[f1 / weights["w"], 0.0, c1x],
                   [0.0, f1 / weights["h"], c1y],
                   [0.0, 0.0, 1.0]])
    H = K1 @ _rotation(rx, ry, rz) @ K0i

    total = float(weights["total"])
    parts = []

    if len(vert[0]):
        parts.append(_family_terms(H, vert[0], np.array([1.0, 0.0]),
                                   weights["w_pic"] * lam_v / total,
                                   weights["w"], weights["h"], weights["mean_length"]))
    if len(horiz[0]):
        parts.append(_family_terms(H, horiz[0], np.array([0.0, 1.0]),
                                   weights["w_pic"] * lam_h / total,
                                   weights["w"], weights["h"], weights["mean_length"]))
    if vp_h2 is not None:
        a = H @ horiz[1]
        b = H @ vp_h2
        l = np.cross(a, b)
        d = _normalize(np.array([-l[1], l[0]]))
        parts.append(np.array([math.sqrt(weights["w_eye"]) *
                               float(d @ np.array([0.0, 1.0]))]))

    parts.append(np.array([math.sqrt(weights["w_f"]) * (f1 / f0 - 1.0)]))

    return np.concatenate(parts)


def _levenberg_marquardt(fun, x0, lo, hi, iters=160):
    """Damped Gauss-Newton with a forward-difference Jacobian."""
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    r = fun(x)
    cost = float(r @ r)
    lam = 1e-3
    step = 1e-6 * np.maximum(1.0, np.abs(x))
    for _ in range(iters):
        if cost < 1e-24:
            break
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            xp = x.copy()
            xp[i] = min(hi[i], xp[i] + step[i])
            J[:, i] = (fun(xp) - r) / (xp[i] - x[i] + _EPS)
        A = J.T @ J
        g = J.T @ r
        accepted = False
        for _try in range(8):
            try:
                dx = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-9)),
                                     -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = np.clip(x + dx, lo, hi)
            rn = fun(xn)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.4, 1e-9)
                accepted = True
                break
            lam *= 8.0
        if not accepted:
            break
        if np.max(np.abs(dx)) < 1e-12:
            break
    return x, cost


def _rotation_from_vps(vp_v, vp_h, K0i):
    """Rectifying rotation from the two vanishing-point directions.

    Rows are (world X, world Y, world Z) expressed in camera coordinates, so
    the rotation maps the world axes onto the image axes.
    """
    y_axis = _normalize(K0i @ vp_v)
    if y_axis[1] < 0:
        y_axis = -y_axis
    x_axis = K0i @ vp_h
    x_axis = x_axis - float(x_axis @ y_axis) * y_axis
    if np.linalg.norm(x_axis) < _EPS:
        return None
    x_axis = _normalize(x_axis)
    if float(np.cross(x_axis, y_axis)[2]) < 0.0:
        x_axis = -x_axis
    z_axis = _normalize(np.cross(x_axis, y_axis))
    return np.vstack([x_axis, y_axis, z_axis])

# ---------------------------------------------------------------------------
# Auto's measured damping
# ---------------------------------------------------------------------------
def _saturate(lam: float) -> float:
    return float(lam / (lam + 1.0))


def correction_fraction_vertical(keystone_rad: float) -> float:
    """Removed fraction of visible vertical convergence (radians), not pitch."""
    return _saturate(L0_V * math.exp(-(keystone_rad ** 2) / (2.0 * SIGMA_V ** 2)))


def correction_fraction_pitch(pitch_rad: float) -> float:
    """Removed fraction of visible vertical convergence, as a function of pitch.

    Piecewise-linear through the corpus bin medians, clamped to [0, 1]. This is the
    shipped Auto response; :func:`correction_fraction_vertical` is the superseded
    keystone-units form, kept for comparison.
    """
    deg = abs(math.degrees(pitch_rad))
    return min(1.0, max(0.0, AUTO_PITCH_GAIN * float(np.interp(
        deg, AUTO_PITCH_KNOTS_DEG, AUTO_PITCH_KNOTS_FRACTION))))


def correction_fraction_horizontal(pitch_rad: float, yaw_rad: float) -> float:
    """Fraction of the horizontal correction Auto applies.

    Gated on pitch: with no pitch the verticals are already parallel and Auto
    leaves the horizontals alone.
    """
    gate = 1.0 - math.exp(-(pitch_rad ** 2) / (2.0 * SIGMA_PITCH_GATE ** 2))
    return _saturate(LAMBDA0_H * math.exp(-(yaw_rad ** 2) / (2.0 * SIGMA_YAW ** 2)) * gate)


def mode_damping(mode: str, rx: float, ry: float, rz: float,
                 *, vertical_keystone_rad=None) -> tuple[float, float, float]:
    """Requested mode response; Auto's first value is a convergence fraction.

    The caller converts that physical fraction to a pitch rotation scale.
    Full/level/vertical are approximations; the Auto numbers are MEASURED.
    Auto's first value is keyed on the UNDAMPED camera pitch (Round 13 corpus
    measurement); the detected keystone only sets the inversion target.
    """
    if mode == "off":
        return 0.0, 0.0, 0.0
    if mode == "level":
        return 0.0, 0.0, ROLL_LEVEL
    if mode == "vertical":
        return 1.0, 0.0, ROLL_LEVEL
    if mode == "full":
        return 1.0, 1.0, 1.0
    if mode == "auto":
        if vertical_keystone_rad is None:
            raise ValueError("Auto damping needs visible vertical convergence, not pitch")
        return (correction_fraction_pitch(rx),
                correction_fraction_horizontal(rx, ry), ROLL_AUTO)
    raise ValueError(f"unknown mode {mode!r}")


def _vertical_convergence(seg_n, w, h, H=None):
    """P95-P5 signed vertical direction spread in pixel angles, in radians.

    CHOICE: observed supports approximate the visible world lines used in the
    measured fit. This transfers the units without consulting scene truth.
    """
    lines = _homogeneous_lines(seg_n)
    if H is not None:
        lines = lines @ np.linalg.inv(H)
    tilt = np.arctan2(-lines[:, 1] * w, lines[:, 0] * h)
    tilt = (tilt + math.pi / 2) % math.pi - math.pi / 2
    return float(np.diff(np.percentile(tilt, [5, 95]))[0])


def _auto_pitch_scale(rotation, fractions, K1, K0i, seg_v, w, h, keystone):
    """Convert a physical convergence-removal fraction to an Euler scale.

    CHOICE: bounded scalar inversion; a rotation fraction is not the measured
    line-angle fraction. Yaw and roll retain their separate measured schedules.
    """
    rx, ry, rz = rotation
    sx, sy, sz = fractions
    target = (1 - sx) * keystone

    def spread(scale):
        H = K1 @ _rotation(rx * scale, ry * sy, rz * sz) @ K0i
        return _vertical_convergence(seg_v, w, h, H)

    low, high = 0.0, 1.0
    if target >= spread(low):
        return low
    if target <= spread(high):
        return high
    for _ in range(32):
        mid = (low + high) / 2
        if spread(mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def _alignment_quality(H, seg_v, seg_h, w, h):
    """Check undamped support scatter and systematic signed pixel-angle bias.

    CHOICE: one degree scatter cap; signed bias plus its support margin must
    be at most .1 degree or strictly below that family's absolute scatter.
    Calibrated on real photographs (evidence/round10.txt), independent of pixel
    dimensions. This checks support consistency, not ground-truth accuracy.
    """
    medians, signed_medians, margins = [], [], []
    for seg, axis in ((seg_v, 0), (seg_h, 1)):
        lines = _homogeneous_lines(seg) @ np.linalg.inv(H)
        directions = _normalize_rows(np.column_stack([-lines[:, 1] * w, lines[:, 0] * h]))
        residual = np.abs(directions[:, axis])
        lengths = np.hypot((seg[:, 2] - seg[:, 0]) * w,
                           (seg[:, 3] - seg[:, 1]) * h)

        def median(values):
            order = np.argsort(values)
            index = np.searchsorted(np.cumsum(lengths[order]), 0.5 * lengths.sum())
            return float(values[order[index]])

        medians.append(median(residual))
        # Orient each line toward its intended positive axis before measuring
        # signed error; absolute medians conflate noisy evidence with bias.
        sign = np.where(directions[:, 1-axis] < 0, -1., 1.)
        angles = np.degrees(np.arcsin(np.clip(directions[:, axis] * sign, -1, 1)))
        centre = median(angles)
        effective_n = lengths.sum()**2 / max(float(lengths @ lengths), _EPS)
        signed_medians.append(centre)
        # 2 * 1.253 (median SE) * 1.483 (Gaussian MAD conversion).
        margins.append(3.72 * median(np.abs(angles-centre)) / math.sqrt(effective_n))
    scale = 4.0 / max(1, max(w, h) - 1)
    median_deg = np.degrees(np.arcsin(np.clip(medians, 0, 1)))
    # MEASURED: 5.0 degrees, not 1.0. On a calibration/holdout split (25 and 27
    # solvable frames) 1.0 accepts 9 and 4 and leaves the holdout median at
    # 3.0150 against an identity of 3.5205; 5.0 accepts 16 and 16 and takes the
    # holdout to 2.3482. The plateau runs 5.0-8.0 on both splits and accept-all
    # collapses to 4.78, so the guard still does work, it was just four degrees
    # too tight. evidence/squarewell.txt section 10.
    limit_deg = 5.0
    bounds = np.abs(signed_medians) + margins
    # Precise families keep the .1-degree guard; photographic scatter supplies
    # a noise budget rather than making uncertainty alone reject every fit.
    bias_limits = np.maximum(0.1, median_deg)
    scatter_passed = bool(np.all(np.isfinite(median_deg)) and max(median_deg) <= limit_deg)
    bias_passed = bool(np.all(np.isfinite(bounds)) and
                       np.all((bounds <= 0.1) | (bounds < median_deg)))
    return dict(family_median_deg=median_deg.tolist(),
                limit_deg=limit_deg,
                # Legacy four-pixel diagnostic only; never an acceptance limit.
                worst_scale=float(max(medians) / scale),
                family_signed_median_deg=signed_medians,
                family_median_margin_deg=margins,
                family_bias_bound_deg=bounds.tolist(), bias_limit_deg=0.1,
                family_bias_limit_deg=bias_limits.tolist(),
                scatter_passed=scatter_passed, bias_passed=bias_passed,
                passed=scatter_passed and bias_passed)


def _source_calibration(seg_v, seg_h, vp_v, vp_h, w, h, prior):
    """Estimate K0 from perpendicular families before freeing destination K1.

    CHOICE: centered, square-pixel source; robust observed-line calibration.
    A free destination centre here would make source focal unidentifiable.
    """
    K = _intrinsics_norm(prior, w, h)
    R = _rotation_from_vps(vp_v, vp_h, np.linalg.inv(K))
    if R is None:
        return prior, dict(accepted=False, reason="degenerate rays")
    segments = np.concatenate([seg_v, seg_h])
    mean_length = np.mean(np.hypot((segments[:, 2]-segments[:, 0])*w,
                                  (segments[:, 3]-segments[:, 1])*h))

    def residual(theta):
        K = _intrinsics_norm(float(np.exp(theta[3])), w, h)
        H = K @ _rotation(*theta[:3]) @ np.linalg.inv(K)
        return np.concatenate([
            _family_terms(H, seg_v, np.array([1., 0.]), 1, w, h, mean_length),
            _family_terms(H, seg_h, np.array([0., 1.]), 1, w, h, mean_length)])

    start = np.r_[_euler_xyz(R), math.log(prior)]
    lo = np.array([-1.2]*3 + [math.log(prior/8)])
    hi = np.array([1.2]*3 + [math.log(prior*8)])
    theta, cost = _levenberg_marquardt(residual, start, lo, hi)
    # Profile out rotation: near-frontal/parallel families carry little focal
    # information. This is a resolution policy, not a statistical confidence CI.
    step = 1e-5
    J = np.column_stack([(residual(theta + np.eye(4)[i]*step)-residual(theta))/step
                         for i in range(4)])
    fit = np.linalg.lstsq(J[:, :3], J[:, 3], rcond=None)[0]
    profile = J[:, 3] - np.sum(J[:, :3]*fit, axis=1)
    resolution = (4.0/(max(w, h)-1))/max(float(np.linalg.norm(profile)), _EPS)
    estimated = float(np.exp(theta[3]))
    interior = bool(np.all(theta > lo+1e-4) and np.all(theta < hi-1e-4))
    # Retain plausible metadata; don't chase subpixel calibration noise.
    accepted = bool(interior and resolution < 0.1 and
                    abs(math.log(estimated/prior)) > max(math.log(1.05), 3*resolution))
    return (estimated if accepted else prior), dict(
        accepted=accepted, estimated_px=estimated, prior_px=prior,
        log_focal_resolution=resolution, cost=cost,
        reason="resolved focal discrepancy" if accepted else "retain prior: unresolved or small discrepancy")


def _base_solve(seg_n, w, h, f0, weights, mode, seed, shared_k=False):
    """Solve the full rectification, then apply the mode's damping."""
    vps = estimate_vanishing_points(seg_n * np.array([w, h, w, h]), w, h, seed=seed)
    # VP fitting works in pixels; the energy transforms normalized coordinates.
    for v in vps:
        vp = v["vp"]
        if abs(vp[2]) > _EPS:
            v["vp"] = np.array([vp[0] / vp[2] / w, vp[1] / vp[2] / h, 1.0])
        else:
            v["vp"] = _normalize(np.array([vp[0] / w, vp[1] / h, 0.0]))
    vertical, horizontal = _classify(vps, seg_n, w, h)
    if vertical is None or not horizontal:
        return None

    K0 = _intrinsics_norm(f0, w, h)
    K0i = np.linalg.inv(K0)

    # Rank camera rays, including VPs at infinity. Using image-origin x/y lets
    # a weak VP near the top edge displace the real lateral family (round1.txt).
    def lateral(v):
        return abs(_normalize(K0i @ v["vp"])[0])

    horizontal.sort(key=lateral, reverse=True)
    # Lock the RANSAC family identities before refining their final supports.
    # Refining weak unused intersections first can change the lateral ranking.
    for family in (vertical, *horizontal[:2]):
        vp = family["vp"] * [w, h, 1]
        fitted = _refine_vp(seg_n[family["inliers"]] * [w, h, w, h], w, h, vp)
        family["vp"] = _normalize(fitted / [w, h, 1])
    vp_h = horizontal[0]["vp"]
    vp_h2 = horizontal[1]["vp"] if len(horizontal) > 1 else None

    seg_v = seg_n[vertical["inliers"]]
    seg_h = seg_n[horizontal[0]["inliers"]]
    calibration = None
    if not shared_k:
        f0, calibration = _source_calibration(seg_v, seg_h, vertical["vp"], vp_h,
                                              w, h, f0)
        K0 = _intrinsics_norm(f0, w, h)
        K0i = np.linalg.inv(K0)

    R0 = _rotation_from_vps(vertical["vp"], vp_h, K0i)
    if R0 is None:
        return None
    eul0 = _euler_xyz(R0)

    vert = (seg_v, vertical["vp"])
    horiz = (seg_h, vp_h)
    # PAPER: Perkins applies to three-ray fork junctions. A pair of infinite
    # support lines crossing inside the frame is not a detected fork.
    # evidence/accuracy_gap.txt: no fork detector is available in this port.

    # One undamped candidate and quality decision for every automatic mode;
    # a mode-specific objective could reject Level while accepting Full.
    lam_v = lam_h = 8.0

    weights = dict(weights)
    # Our energy weighting choice: mean-normalised lengths are recovered in the
    # detector, but transferring them to this objective is not proven.
    lengths = np.hypot((seg_n[:, 2] - seg_n[:, 0]) * w,
                       (seg_n[:, 3] - seg_n[:, 1]) * h)
    weights.update(w=w, h=h, total=float(len(seg_n)), mean_length=float(lengths.mean()))

    x0 = np.array([eul0[0], eul0[1], eul0[2], math.log(f0), 0.5, 0.5])
    lo = np.array([-1.2, -1.2, -1.2, math.log(f0) - math.log(8.0), 0.2, 0.2])
    hi = np.array([1.2, 1.2, 1.2, math.log(f0) + math.log(8.0), 0.8, 0.8])
    # Translation leaves every angle term invariant. Without a framing term,
    # optimizing K1's centre only integrates finite-difference roundoff.
    lo[4:] = hi[4:] = 0.5
    if shared_k:
        # Collapse the model to K * R * K^-1: one camera, centre fixed.
        lo[3] = hi[3] = math.log(f0)
        lo[4] = hi[4] = lo[5] = hi[5] = 0.5

    def fun(theta):
        return _energy_residuals(theta, K0, K0i, vert, horiz, vp_h2,
                                 f0, weights, lam_v, lam_h)

    theta, _ = _levenberg_marquardt(fun, x0, lo, hi)
    rx, ry, rz, log_f1, c1x, c1y = theta
    f1 = float(np.exp(log_f1))
    K1 = np.array([[f1 / w, 0.0, c1x], [0.0, f1 / h, c1y], [0.0, 0.0, 1.0]])

    quality = _alignment_quality(K1 @ _rotation(rx, ry, rz) @ K0i, seg_v, seg_h, w, h)

    keystone = _vertical_convergence(seg_v, w, h)
    sx, sy, sz = mode_damping(mode, rx, ry, rz, vertical_keystone_rad=keystone)
    if mode == "auto":
        sx = _auto_pitch_scale((rx, ry, rz), (sx, sy, sz), K1, K0i,
                               seg_v, w, h, keystone)
    R = _rotation(rx * sx, ry * sy, rz * sz)
    H = K1 @ R @ K0i
    if abs(H[2, 2]) < _EPS:
        return None
    return H / H[2, 2], dict(rotation=(rx, ry, rz), focal_px=f1, K1=K1,
                             damping=(sx, sy, sz), vps=vps, quality=quality,
                             source_focal_px=f0, calibration=calibration,
                             vertical_keystone_rad=keystone)


def _family_vp(seg_n: np.ndarray):
    """Least-squares vanishing point of a small family, in normalized coords."""
    if len(seg_n) < 2:
        return None
    lines = _homogeneous_lines(seg_n)
    _, _, vt = np.linalg.svd(lines)
    vp = _normalize(vt[-1])
    if vp[2] < 0:
        vp = -vp
    return vp if np.linalg.norm(vp[:2]) > _EPS else None


def _guided_residuals(theta, K0, K0i, seg_n, axes, w, h):
    rx, ry, rz, log_f1 = theta
    f1 = float(np.exp(log_f1))
    K1 = np.array([[f1 / w, 0.0, 0.5], [0.0, f1 / h, 0.5], [0.0, 0.0, 1.0]])
    H = K1 @ _rotation(rx, ry, rz) @ K0i
    HiT = np.linalg.inv(H).T
    l = _homogeneous_lines(seg_n) @ HiT.T          # l' = H^-T l
    d = _normalize_rows(np.column_stack([-l[:, 1], l[:, 0]]))
    perp = np.where(axes[:, None] == 0, np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    return np.sum(d * perp, axis=1)


def _guided_solve(seg_n, w, h, f0):
    """Solve four guide segments as hard constraints (residual -> 0).

    Each segment is classified by its pre-correction orientation: a mostly
    vertical guide becomes a *vertical* constraint, a mostly horizontal one a
    *horizontal* constraint.  Four residuals against four unknowns (three
    rotation angles and the destination focal); the destination principal point
    stays at the image centre and the scale is normalized out.
    """
    if len(seg_n) != 4:
        raise ValueError("guided mode needs exactly 4 segments")
    # CHOICE: compare in pixels. seg_n is normalized per axis, so the test is
    # aspect-biased and misclassifies guides on very wide or tall frames.
    dx = np.abs(seg_n[:, 2] - seg_n[:, 0]) * w
    dy = np.abs(seg_n[:, 3] - seg_n[:, 1]) * h
    axes = np.where(dy >= dx, 0, 1)          # 0 = vertical target, 1 = horizontal
    if len(np.unique(axes)) < 2:
        raise ValueError("guided mode needs at least one vertical and one "
                         "horizontal guide segment")

    K0 = _intrinsics_norm(f0, w, h)
    K0i = np.linalg.inv(K0)

    # Start from the vanishing-point geometry when it is recoverable, otherwise
    # from identity.  Starting at identity converges to a second, much flatter
    # zero with a compensating focal, which satisfies the guides but does not
    # recover the camera.
    start = np.zeros(3)
    # Four guides give only two lines per family, below the RANSAC minimum of
    # three inliers, so fit each family's VP directly by TLS instead.
    vp_v = _family_vp(seg_n[axes == 0])
    vp_h = _family_vp(seg_n[axes == 1])
    if vp_v is not None and vp_h is not None:
        R0 = _rotation_from_vps(vp_v, vp_h, K0i)
        if R0 is not None:
            start = _euler_xyz(R0)
    x0 = np.array([start[0], start[1], start[2], math.log(f0)])
    lo = np.array([-1.0, -1.0, -1.0, math.log(f0) - math.log(6.0)])
    hi = np.array([1.0, 1.0, 1.0, math.log(f0) + math.log(6.0)])

    def fun(theta):
        return _guided_residuals(theta, K0, K0i, seg_n, axes, w, h)

    theta, _ = _levenberg_marquardt(fun, x0, lo, hi, iters=300)
    rx, ry, rz, log_f1 = theta
    f1 = float(np.exp(log_f1))
    K1 = np.array([[f1 / w, 0.0, 0.5], [0.0, f1 / h, 0.5], [0.0, 0.0, 1.0]])
    H = K1 @ _rotation(rx, ry, rz) @ K0i
    if abs(H[2, 2]) < _EPS:
        raise RuntimeError("guided solve degenerated")
    H = H / H[2, 2]
    resid = np.abs(fun(theta))
    return H, dict(rotation=(rx, ry, rz), focal_px=f1, K1=K1,
                   damping=(1.0, 1.0, 1.0),
                   guide_residual_deg=np.degrees(np.arcsin(np.clip(resid, 0, 1))))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def frame_output(H: np.ndarray) -> np.ndarray:
    """Centre Adobe's 90% inner / 10% outer corrected-quad bounds.

    Acts on normalized coordinates after solving; changes translation only.
    The weights reproduce all 16 active adobe_truth slots to stored precision.
    PerspectiveScale=100 does not identify the destination focal/zoom policy.
    """
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)) or H[2, 2] == 0:
        raise ValueError("framing needs a finite 3x3 homography with nonzero H22")
    H = H / H[2, 2]
    corners = np.array([[0., 0., 1.], [1., 0., 1.],
                        [1., 1., 1.], [0., 1., 1.]]) @ H.T
    if np.min(corners[:, 2]) <= 0:
        raise ValueError("framing crosses the projective horizon")
    bounds = np.sort(corners[:, :2] / corners[:, 2:], axis=0)
    centre = (bounds[0] + 9*bounds[1] + 9*bounds[2] + bounds[3]) / 20
    translation = np.eye(3)
    translation[:2, 2] = 0.5 - centre
    return translation @ H


def scale_output(H: np.ndarray) -> np.ndarray:
    """Normalize destination size with the measured 10% boundary-fit switch.

    Preserve the 90/10 bounds centre; this commutes with :func:`frame_output`.
    Use the scale covering the original frame if it is at most 1.10 times the
    scale containing the inner bounds; otherwise use the inner-bounds scale.
    MEASURED: evidence/round8.txt; approximate, with retained native exceptions.
    """
    framed = frame_output(H)
    H = np.asarray(H, dtype=float)
    H = H / H[2, 2]
    corners = np.array([[0., 0., 1.], [1., 0., 1.],
                        [1., 1., 1.], [0., 1., 1.]]) @ framed.T
    q = corners[:, :2] / corners[:, 2:] - 0.5
    edges = np.roll(q, -1, axis=0) - q
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    distances = np.sum(normals * q, axis=1)
    if np.any(distances <= 0):
        raise ValueError("scaling needs a convex quad enclosing its bounds centre")
    cover = float(np.max(np.sum(np.abs(normals), axis=1) / (2*distances)))
    # An exact 45-degree diamond has both inner order statistics on the centre,
    # so the inner scale is .5/0. Fall back to cover there; the limit of the
    # ratio is 0, so cover is also what the rule would select.
    inner_extent = float(np.max(np.abs(np.sort(q, axis=0)[1:3])))
    if inner_extent > _EPS:
        inner = 0.5 / inner_extent
        # The branch is geometric, shared by every mode; do not tune damping.
        scale = cover if cover <= 1.10*inner else inner
    else:
        scale = cover
    centre = 0.5 - (framed[:2, 2] - H[:2, 2])
    transform = np.diag([scale, scale, 1.])
    transform[:2, 2] = (1-scale)*centre
    return transform @ H


def solve_ex(
    segments=None,
    w: int | None = None,
    h: int | None = None,
    focal_px: float | None = None,
    mode: str = "auto",
    *,
    image=None,
    max_side: int = DEFAULT_MAX_SIDE,
    damping=None,
    seed: int = 0,
    shared_k: bool = False,
    framing: bool = False,
    scaling: bool = False,
) -> dict:
    """Like :func:`solve` but returns a diagnostic dictionary.

    Keys: ``H``, ``mode``, ``segments``, ``focal_px``, ``rotation``,
    ``damping``, ``vps``, ``guide_residual_deg``, ``quality``, ``ok``, ``reason``.
    ``focal_px`` retains the supplied value; ``source_focal_px`` reports the
    source focal actually used. ``calibration`` describes the geometry check
    in automatic modes (None for guided/off and shared_k comparisons).
    ``vertical_keystone_rad`` is the detected P95-P5 vertical direction spread.
    ``damping`` reports applied Euler scales, including Auto's converted pitch
    scale, rather than its requested physical convergence-removal fraction.
    Automatic modes check undamped raw support scatter and signed median
    pixel-angle bias with a MAD-based support margin. These checks establish
    support consistency, not a guarantee of world-camera correctness.
    ``framing=True`` applies :func:`frame_output` after the solve and explicit
    damping override; ``H_unframed`` then retains the original mapping.
    ``scaling=True`` applies :func:`scale_output` next; ``H_unscaled`` retains
    its input. Both options default to False and leave the geometry unchanged.
    ``quality`` is None for
    guided/off or when no fit is available. Failure returns identity.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if segments is None:
        if image is None:
            raise ValueError("pass segments= or image=")
        if w is None or h is None:
            h, w = np.asarray(image).shape[:2]
        segments = detect_segments(image, max_side=max_side)
    else:
        segments = np.asarray(segments, dtype=float).reshape(-1, 4)
        if w is None or h is None:
            if image is None:
                raise ValueError("pass w= and h= when segments are supplied")
            h, w = np.asarray(image).shape[:2]
    w, h = int(w), int(h)

    out = dict(H=np.eye(3), mode=mode, segments=segments, focal_px=None,
               rotation=None, damping=None, vps=[], guide_residual_deg=None,
               quality=None, source_focal_px=None, calibration=None, ok=False, reason="")
    if mode == "off":
        out.update(ok=True, reason="mode off")
        return out
    if mode == "guided" and len(segments) != 4:
        raise ValueError("guided mode needs exactly 4 segments")
    if len(segments) < 4:
        out["reason"] = f"only {len(segments)} segments"
        return out

    if focal_px is None:
        focal_px = metadata_focal_px(None, w, h)
    f0 = float(focal_px)
    if not np.isfinite(f0) or f0 <= 0.0:
        out["reason"] = "bad focal"
        return out

    seg_n = segments / np.array([w, h, w, h], dtype=float)
    out["focal_px"] = f0

    try:
        if mode == "guided":
            H, info = _guided_solve(seg_n, w, h, f0)
        else:
            got = _base_solve(seg_n, w, h, f0, _FULL_WEIGHTS, mode, seed,
                              shared_k=shared_k)
            if got is None:
                out["reason"] = "insufficient line structure"
                return out
            H, info = got
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        out["reason"] = str(exc)
        return out

    out.update(rotation=info["rotation"], damping=info["damping"],
               vertical_keystone_rad=info.get("vertical_keystone_rad"),
               vps=info.get("vps", []), quality=info.get("quality"),
               source_focal_px=info.get("source_focal_px", f0),
               calibration=info.get("calibration"))
    if out["quality"] is not None and not out["quality"]["passed"]:
        out["reason"] = "poor undamped family alignment"
        return out

    if damping is not None:
        d = (float(damping),) * 3 if np.isscalar(damping) else tuple(map(float, damping))
        if len(d) != 3:
            raise ValueError("damping must be a scalar or a 3-tuple")
        rx, ry, rz = info["rotation"]
        K0i = np.linalg.inv(_intrinsics_norm(out["source_focal_px"], w, h))
        K1 = info["K1"]
        H = K1 @ _rotation(rx * d[0], ry * d[1], rz * d[2]) @ K0i
        H = H / H[2, 2]
        info["damping"] = d

    if not np.all(np.isfinite(H)):
        out["reason"] = "non-finite homography"
        return out

    # A homography can be finite and still put frame corners on or across the
    # projective horizon; the quad then wraps and the output cannot be sampled
    # as a frame. frame_output makes this exact test, but framing is optional,
    # so without it a bad map used to be returned as ok=True.
    _corner_w = (np.array([[0., 0., 1.], [1., 0., 1.],
                           [1., 1., 1.], [0., 1., 1.]]) @ H.T)[:, 2]
    if np.min(_corner_w) <= 0:
        out["reason"] = "crosses the projective horizon"
        return out

    if framing:
        try:
            framed = frame_output(H)
        except ValueError as exc:
            out["reason"] = str(exc)
            return out
        out["H_unframed"] = H
        H = framed

    if scaling:
        try:
            scaled = scale_output(H)
        except ValueError as exc:
            out["reason"] = str(exc)
            return out
        out["H_unscaled"] = H
        H = scaled

    out.update(H=H, ok=True, rotation=info["rotation"],
               damping=info["damping"], vps=info.get("vps", []),
               guide_residual_deg=info.get("guide_residual_deg"),
               reason="ok")
    return out


def solve(segments=None, w=None, h=None, focal_px=None, mode="auto", **kwargs):
    """Return the 3x3 rectifying homography in normalized coordinates.

    Parameters
    ----------
    segments : (N, 4) array, optional
        ``x1, y1, x2, y2`` pixel segments.  Omit and pass ``image=`` to use the
        bundled detector, or pass your own detector's output.
    w, h : int
        Image size in pixels.  Required with ``segments``; inferred from
        ``image`` otherwise.
    focal_px : float, optional
        Source focal length in pixels.  Defaults to a 35 mm full-frame
        equivalent for the image size. Automatic modes check this prior against
        robust perpendicular-family geometry; unresolved/small discrepancies
        retain it. Guided and shared_k use it unchanged. Only metadata goes
        through :func:`metadata_focal_px`; geometric estimates are not clamped.
    mode : {'off', 'level', 'vertical', 'full', 'auto', 'guided'}
        ``guided`` requires exactly four segments.
    image : ndarray, optional
        Grayscale (or BGR) image; only used when ``segments`` is None.
    max_side : int
        Working resolution for the detector.
    damping : float or 3-tuple, optional
        Override the per-mode rotation damping ``(rx, ry, rz)``.  ``1.0``
        disables damping entirely and returns the undamped solve.
    seed : int
        RANSAC seed.
    shared_k : bool
        Collapse the model to the single-camera ``K * R * K**-1`` for
        comparison.  Only meaningful as a sanity check.
    framing : bool
        Centre the corrected quad with :func:`frame_output` after solving.
        Defaults to False; changes only output translation, with no zoom/crop.
    scaling : bool
        Normalize destination size with :func:`scale_output` after framing.
        Defaults to False. Preserves the bounds centre and all output angles.

    Returns
    -------
    (3, 3) float array
        Homography acting on normalized coordinates.  Identity on failure or
        when the mode is ``off``.
    """
    return solve_ex(segments, w, h, focal_px, mode, **kwargs)["H"]
