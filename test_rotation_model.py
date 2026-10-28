"""
Decisive model test: is Adobe's shipped Upright transform a *pure rotation*
homography H = K R K^-1 (3 rotation DOF + 1 focal DOF), or a general 7-parameter
projective fit?

The unpublished CVPR-2012 paper describes a 7-parameter model. Earlier binary and
third-party evidence suggested the shipped code instead emits K R K^-1 with an
estimated focal length. This fits the rotation model to Adobe's real output and
reports the residual, over the whole harvested corpus.

If the residual is at float noise, the shipped model is K R K^-1 and the paper's
extra DOF are unused in production.
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import least_squares

IDENT = np.eye(3)
MODEL = os.environ.get('MODEL', 'freeK')


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def to_pixel(Hn, w, h):
    D = np.diag([float(w), float(h), 1.0])
    return D @ Hn @ np.linalg.inv(D)


def rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def _prep(Hp, w, h):
    yy, xx = np.mgrid[0:h:max(1, h // 12), 0:w:max(1, w // 12)]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1).astype(float)
    q = (Hp @ p.T).T
    qn = q[:, :2] / np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])[:, None]
    return p, qn, np.hypot(w, h)


def fit_rotation(Hp, w, h, model="freeK"):
    """Fit s * K1 R K0^-1 to Hp. Returns (rms_residual_normalised, angles_deg, f)."""
    p, qn, diag = _prep(Hp, w, h)
    d = 0.5 * (w + h)
    c0 = np.array([w / 2.0, h / 2.0])
    c1 = np.array([w / 2.0, h / 2.0])

    def build(x):
        if model == "sharedK":
            f0 = f1 = x[0]; rx, ry, rz, logs = x[1:]
            a0, a1 = c0, c1
        else:
            f0, rx, ry, rz, logs, dx1, dy1 = x
            f1 = f0
            a0, a1 = c0, c1 + np.array([dx1, dy1])
        K0 = np.array([[f0, 0, a0[0]], [0, f0, a0[1]], [0, 0, 1.0]])
        K1 = np.array([[f1, 0, a1[0]], [0, f1, a1[1]], [0, 0, 1.0]])
        return np.exp(np.clip(logs, -5, 5)) * (K1 @ rot(rx, ry, rz) @ np.linalg.inv(K0))

    def resid(x):
        r = (build(x) @ p.T).T
        rn = r[:, :2] / np.where(np.abs(r[:, 2]) < 1e-12, 1e-12, r[:, 2])[:, None]
        return ((rn - qn) / diag).ravel()

    x0 = (np.array([0.7 * d, 0.0, 0.0, 0.0, 0.0])
          if model == "sharedK" else np.array([0.7 * d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    s = least_squares(resid, x0, max_nfev=400)
    rms = float(np.sqrt(np.mean(resid(s.x) ** 2)))
    f0 = s.x[0]
    rx, ry, rz = s.x[1:4]
    return rms, np.degrees([rx, ry, rz]), f0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lr_all.jsonl"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    recs = load(path)[:limit]
    print(f"testing {len(recs)} photos with model={MODEL}\n")
    print(f"  {'slot':<9} {'n':>5} {'median rms':>12} {'p95 rms':>12} {'max rms':>12}  {'<1e-4':>7}")
    for slot in (1, 2, 3, 4):
        rs = []
        for r in recs:
            v = r["transform"].get(str(slot))
            if v is None:
                continue
            Hn = np.array(v, dtype=float).reshape(3, 3)
            if np.max(np.abs(Hn - IDENT)) < 1e-9:
                continue
            Hp = to_pixel(Hn, r["width"], r["height"])
            try:
                rms, ang, f = fit_rotation(Hp, r["width"], r["height"], MODEL)
            except Exception:
                continue
            rs.append(rms)
        if not rs:
            continue
        rs = np.array(rs)
        print(f"  {slot:<9} {len(rs):5d} {np.median(rs):12.3e} "
              f"{np.percentile(rs,95):12.3e} {rs.max():12.3e}  {100.0*np.mean(rs<1e-4):6.1f}%")
    print("\n(rms is in units of the image diagonal; <1e-4 means sub-pixel agreement)")


if __name__ == "__main__":
    main()
