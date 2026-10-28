"""
Mode-accurate Upright, matched to the *shipped* Adobe build.

`upright_core.upright_transform` lets the optimiser satisfy every family
residual regardless of the adaptive weights, so all four modes collapse onto the
same homography. Measurements against real Lightroom (§3.4/§3.5 of the report)
show the shipped behaviour is instead:

  * solve the full rectification once,
  * then apply a *mode-dependent rotation damping*,
  * where `Auto`'s keystone damping follows a curve recovered from 7,256 real
    Adobe solves, while roll is damped separately and only slightly.

Two structural facts from the measurements are honoured here:

  1. the homography is  H = s * K1 * R * K0^-1  with K1 != K0, not K R K^-1;
  2. `Auto` is not "less of everything" -- it levels the horizon almost
     unconditionally (roll ratio 0.928) and damps *keystone* hard (0.262).

The damping constants below are measured, not derived; `corpus_lambda_curve.py`
regenerates them.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import rq
from scipy.spatial.transform import Rotation as Rot

import upright_core as uc

# --- measured constants ----------------------------------------------------
# Fitted from the shipped solver driving four controlled images whose camera
# rotation is known exactly, plus 7,256 real Adobe solves.  Both weights are
# functions of the SAME estimated rotation magnitude theta, and the recovered
# sigmas reproduce the CVPR-2012 paper's published constants:
#     sigma_v = 15.47 deg  vs paper pi/12 = 15.00 deg   (3.1% off)
#     sigma_h = 13.47 deg  vs paper pi/15 = 12.00 deg  (12.2% off)
# The response is saturating: correction_fraction = lambda / (lambda + 1), which
# is why `Auto` is never a full rectification even on a level photo.
# The two sigmas are ALSO present as float32 literals in the shipping binary
# (CameraRaw.lrtoolkit/.../CameraRaw) in both sigma^2 and 1/(2 sigma^2) forms,
# at exactly the paper's values. Those literals are the ground truth; the
# behavioural fit below is an inference through an assumed response curve and
# runs 3%/12% high. Use the literals.
# --- AUTO's two weights, both measured on controlled-angle scenes -------------
# 25 scenes with exactly known rotation, solved by the real Lightroom.
# VERTICAL: decays with the PITCH magnitude (7-point sweep; half-effect at ~26deg).
# HORIZONTAL: gated on pitch being non-zero -- on a pure-yaw frame Auto is a
#   no-op (moves horizontal convergence by 0.01deg while Full removes 96%),
#   and only once pitch is present does it correct, damped by the YAW magnitude.
#   Fitted: L0h=76.7, sigma_yaw=11.6deg, sigma_gate=22.1deg.
#   sigma_yaw=11.6deg is a 3.3% match to the paper's sigma_h = pi/15 = 12.0deg,
#   and to the float32 literal 1/(2 sigma_h^2)=11.3986332 in CameraRaw.
LAMBDA0_V = 2.64   # n=80 fit [1.46,4.65]; asymptote 0.725 (was 10.12, far too aggressive)
SIGMA_V = np.radians(13.33)   # keystone-angle units, 95% CI 9.94-22.72, brackets pi/12 (n=80)
# NOTE: fitting against CAMERA PITCH instead gives 22.26 deg [16.67,39.58]. The literal is
# bracketed in keystone units and excluded in pitch units; the binary does not say which.
LAMBDA0_H = 76.73
SIGMA_YAW = np.radians(11.60)
SIGMA_PITCH_GATE = np.radians(22.10)

# paper / binary literals, exposed for A/B testing
SIGMA_V_LITERAL = np.pi / 12
SIGMA_H_LITERAL = np.pi / 15

def _saturate(lam: float) -> float:
    return float(lam / (lam + 1.0))


def correction_fraction(pitch_rad: float) -> float:
    """Fraction of the VERTICAL (pitch) correction `Auto` actually applies."""
    return _saturate(LAMBDA0_V * np.exp(-(pitch_rad ** 2) / (2.0 * SIGMA_V ** 2)))


def correction_fraction_h(pitch_rad: float, yaw_rad: float) -> float:
    """Fraction of the HORIZONTAL (yaw) correction `Auto` applies.

    Gated on pitch: with no pitch the frame's verticals are already parallel, so
    Auto does essentially nothing at all -- including leaving yaw-induced
    horizontal convergence untouched.
    """
    gate = 1.0 - np.exp(-(pitch_rad ** 2) / (2.0 * SIGMA_PITCH_GATE ** 2))
    return _saturate(LAMBDA0_H * np.exp(-(yaw_rad ** 2) / (2.0 * SIGMA_YAW ** 2)) * gate)


def _split(H: np.ndarray, K: np.ndarray):
    R = uc.polar_rot(np.linalg.inv(K) @ H @ K) if hasattr(uc, "polar_rot") else _polar(
        np.linalg.inv(K) @ H @ K)
    return Rot.from_matrix(R).as_euler("xyz")


def _polar(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def _mode_scales(mode: str, rx: float, ry: float):
    """-> (scale_rx, scale_ry, scale_rz) for the requested mode."""
    theta = float(np.hypot(rx, ry))          # estimated rotation magnitude
    if mode == "level":
        return 0.0, 0.0, ROLL_OTHER
    if mode == "vertical":
        return KEYSTONE_VERTICAL, KEYSTONE_VERTICAL, ROLL_OTHER
    if mode == "full":
        return KEYSTONE_FULL, KEYSTONE_FULL, ROLL_OTHER
    if mode == "auto":
        # anisotropic and saturating, exactly as measured: horizontals are
        # weighted ~8x more strongly than verticals, which is why real `Auto`
        # keeps 100% of the crop while still leaving verticals converging.
        return correction_fraction(rx), correction_fraction_h(rx, ry), ROLL_AUTO
    raise ValueError(mode)


def upright_matched(gray, f0=None, mode="auto", max_side=1200, verbose=False):
    """Drop-in replacement for upright_core.upright_transform, mode-accurate."""
    base = uc.upright_transform(gray, f0=f0, mode="full", max_side=max_side,
                                verbose=verbose)
    H = np.array(base["H"], dtype=float)
    w = gray.shape[1]
    h = gray.shape[0]
    f = base.get("f0") or f0 or (1.2 * max(w, h))  # clustering-stage assumption f ~ maxdim
    K = uc.intrinsics(f, f, w / 2.0, h / 2.0)

    # Recover the two-camera structure H = s*K1*R*K0^-1 from the base solve, so
    # damping the rotation preserves K1 != K0 instead of collapsing to K R K^-1.
    M = np.linalg.inv(K) @ H @ K
    R_old = _polar(M)
    eul = Rot.from_matrix(R_old).as_euler("xyz")
    sx, sy, sz = _mode_scales(mode, eul[0], eul[1])
    eul_d = np.array([eul[0] * sx, eul[1] * sy, eul[2] * sz])
    R_new = Rot.from_euler("xyz", eul_d).as_matrix()

    # K1 is whatever the base solve's output camera was: H*K0*R_old^-1 = s*K1,
    # upper-triangular, so take its RQ factor.
    K1 = H @ K @ np.linalg.inv(R_old)      # = s * K1, already near-triangular
    for i in range(3):
        if K1[i, i] < 0:
            K1[i] *= -1.0
    K1 = K1 / K1[2, 2]

    Ki = np.linalg.inv(K)
    base_n = K1 @ R_old @ Ki
    base_n = base_n / base_n[2, 2]
    t = (H / H[2, 2])[:2, 2] - base_n[:2, 2]      # base solve's extra translation

    Hd = K1 @ R_new @ Ki
    Hd = Hd / Hd[2, 2]
    Hd[:2, 2] += t

    out = dict(base)
    out["H"] = Hd
    out["mode"] = mode
    out["damped"] = (sx, sy, sz)
    out["scales"] = (sx, sy, sz)
    return out
