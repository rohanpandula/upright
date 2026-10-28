"""Adversarial fuzz harness for research/upright/port/upright_port.py (MIT, read-only).

Run: python3.12 evidence/port_fuzz_harness.py
Writes JSON to evidence/port_fuzz_raw.json for the report writer.
"""
from __future__ import annotations

import json
import math
import signal
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/Users/rohan/Projects/trichroma/research/upright/evidence")
import port_fuzz_snapshot as up  # noqa: E402

W, H = 1600, 1200
ALARM = 60
RESULTS = []


class Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise Timeout(f"exceeded {ALARM}s")


signal.signal(signal.SIGALRM, _alarm)


def project_scene(pitch_deg, yaw_deg, roll_deg, focal_px, w=W, h=H):
    R = up._rotation(math.radians(pitch_deg), math.radians(yaw_deg),
                     math.radians(roll_deg))
    Rt = R.T

    def proj(p):
        q = Rt @ np.asarray(p, dtype=float)
        if q[2] <= 0.5:
            return None
        return np.array([focal_px * q[0] / q[2] + w / 2.0,
                         focal_px * q[1] / q[2] + h / 2.0])

    segs, kinds = [], []
    for X in (-3.0, -1.5, 0.0, 1.5, 3.0):
        for Z in (5.0, 7.0):
            a, b = proj([X, -2.0, Z]), proj([X, 3.0, Z])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                segs.append([*a, *b])
                kinds.append(0)
    for Y in (-1.5, 0.0, 1.5):
        for Z in (5.0, 7.0):
            a, b = proj([-4.0, Y, Z]), proj([4.0, Y, Z])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                segs.append([*a, *b])
                kinds.append(1)
    for Y in (-1.0, 1.0):
        for X in (-2.0, 2.0):
            a, b = proj([X, Y, 4.0]), proj([X, Y, 9.0])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                segs.append([*a, *b])
                kinds.append(2)
    return np.asarray(segs, dtype=float), R, np.asarray(kinds)


BASE_SEG, BASE_R, BASE_KIND = project_scene(12.0, 0.0, 3.0, 900.0)
# A clean guided quartet: two verticals, two horizontals.
GUIDED4 = np.array([[700.0, 200.0, 700.0, 1000.0],
                    [900.0, 250.0, 900.0, 1050.0],
                    [200.0, 400.0, 1400.0, 420.0],
                    [150.0, 800.0, 1450.0, 780.0]])


def assess(H):
    """Classify an ok=True homography as usable or silently broken."""
    bad = []
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        return ["shape %r" % (H.shape,)], {}
    if not np.all(np.isfinite(H)):
        bad.append("non-finite H")
    h22 = float(H[2, 2])
    if not np.isfinite(h22) or h22 == 0.0:
        bad.append("H22 zero/non-finite")
    if np.all(np.isfinite(H)):
        s = np.linalg.svd(H, compute_uv=False)
        if not s[-1] > 1e-12 * s[0]:
            bad.append("singular/rank-deficient")
        else:
            if s[0] / s[-1] > 1e12:
                bad.append("condition number %.3g" % (s[0] / s[-1]))
    corners = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
    P = corners @ H.T
    z = P[:, 2]
    if not np.all(np.isfinite(z)):
        bad.append("non-finite corner w")
    elif np.any(z == 0.0):
        bad.append("corner on horizon")
    elif not (np.all(z > 0) or np.all(z < 0)):
        bad.append("corners straddle horizon")
    elif np.all(np.isfinite(P)):
        xy = P[:, :2] / P[:, 2:3]
        if not np.all(np.isfinite(xy)):
            bad.append("non-finite corner xy")
        else:
            area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                             - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
            if area < 1e-12:
                bad.append("degenerate corner quad (area 0)")
            if np.max(np.abs(xy)) > 1e6:
                bad.append("extreme corner magnitude %.3g" % np.max(np.abs(xy)))
            return bad, dict(area=float(area), maxabs=float(np.max(np.abs(xy))),
                             h22=h22, cond=float(np.linalg.cond(H)) if np.all(np.isfinite(H)) else None)
    return bad, dict(h22=h22)


def run(case, fn, *a, expect=None, **kw):
    rec = dict(case=case, expect=expect, args=repr(a)[:400], kw=repr(kw)[:200])
    t0 = time.perf_counter()
    signal.alarm(ALARM)
    try:
        out = fn(*a, **kw)
        rec["outcome"] = "returned"
        if isinstance(out, dict):
            rec["ok"] = bool(out.get("ok"))
            rec["reason"] = str(out.get("reason"))[:200]
            if out.get("ok"):
                bad, stats = assess(out.get("H"))
                rec["bad"] = bad
                rec["stats"] = stats
                rec["verdict"] = "SILENT_BAD" if bad else "ok"
            else:
                rec["verdict"] = "graceful_fail"
        else:
            rec["ok"] = None
            bad, stats = assess(out)
            rec["bad"] = bad
            rec["stats"] = stats
            rec["verdict"] = "SILENT_BAD(solve)" if bad else "ok(solve)"
    except Timeout as e:
        rec["outcome"] = "timeout"
        rec["exc"] = str(e)
        rec["verdict"] = "HANG"
    except Exception as e:  # noqa: BLE001
        rec["outcome"] = "raise"
        rec["exc"] = f"{type(e).__name__}: {e}"
        rec["tb"] = traceback.format_exc().splitlines()[-4:]
        rec["verdict"] = "CRASH"
    finally:
        signal.alarm(0)
    rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    RESULTS.append(rec)
    print(f"{rec['verdict']:16s} {rec['ms']:9.1f}ms {case}"
          + (f"  <{rec.get('exc')}>" if rec.get("exc") else "")
          + (f"  bad={rec.get('bad')}" if rec.get("bad") else ""))
    return rec


def S(**kw):
    kw.setdefault("w", W)
    kw.setdefault("h", H)
    kw.setdefault("focal_px", 900.0)
    return kw


# ---------------------------------------------------------------- A segments
def a_segments():
    ident4 = np.tile([500., 500., 500., 500.], (4, 1))
    ident8 = np.tile([500., 500., 500., 500.], (8, 1))
    coll_h4 = np.array([[100., 400., 1500., 400.]] * 4)
    coll_v4 = np.array([[800., 100., 800., 1100.]] * 4)
    two_pts = np.array([[400., 300., 1200., 900.]] * 4)
    nan_all = BASE_SEG.copy()
    nan_all[:] = np.nan
    nan_one = BASE_SEG.copy()
    nan_one[3, 1] = np.nan
    inf_one = BASE_SEG.copy()
    inf_one[5, 2] = np.inf
    huge = BASE_SEG * 1e6
    neg = BASE_SEG - np.array([2000., 2000., 2000., 2000.])
    for mode in ("auto", "level", "full", "vertical"):
        for name, seg, n in (
            ("identical4", ident4, 4), ("identical8", ident8, 8),
            ("collinear_h4", coll_h4, 4), ("collinear_h8", np.tile(coll_h4[0], (8, 1)), 8),
            ("collinear_v4", coll_v4, 4), ("twopoint4", two_pts, 4),
            ("zero_len4", ident4, 4), ("one_seg", BASE_SEG[:1], 1),
            ("two_seg", BASE_SEG[:2], 2), ("three_seg", BASE_SEG[:3], 3),
            ("nan_all", nan_all, len(nan_all)), ("nan_one", nan_one, len(nan_one)),
            ("inf_one", inf_one, len(inf_one)),
            ("huge_coords", huge, len(huge)), ("negative_coords", neg, len(neg)),
            ("int_dtype", BASE_SEG.astype(np.int64), len(BASE_SEG)),
        ):
            if mode in ("auto", "level") or name in ("identical4", "collinear_h4", "nan_all", "huge_coords"):
                run(f"A/{mode}/{name}", up.solve_ex, seg, mode=mode, **S())
                if mode == "auto" and name in ("identical4", "collinear_h4", "nan_all", "huge_coords"):
                    run(f"A/solve/{mode}/{name}", up.solve, seg, mode=mode, **S())
    # malformed shapes / dtypes
    run("A/shape/(N,3)", up.solve_ex, np.zeros((4, 3)), mode="auto", **S())
    run("A/shape/1d4", up.solve_ex, np.zeros(4), mode="auto", **S())
    run("A/shape/empty", up.solve_ex, np.zeros((0, 4)), mode="auto", **S())
    run("A/shape/None_w_h", up.solve_ex, BASE_SEG, mode="auto")
    run("A/dtype/object", up.solve_ex, BASE_SEG.astype(object), mode="auto", **S())
    run("A/dtype/str", up.solve_ex, BASE_SEG.astype(str), mode="auto", **S())


# ------------------------------------------------------------------ B sizes
def b_sizes():
    mini = np.array([[0.10, 0.10, 0.10, 0.90], [0.30, 0.10, 0.30, 0.90],
                     [0.55, 0.10, 0.55, 0.90], [0.10, 0.40, 0.90, 0.42],
                     [0.10, 0.60, 0.90, 0.58], [0.10, 0.80, 0.90, 0.79]])
    for w, h in ((1, 1), (2, 2), (8, 8), (10000, 3), (3, 10000), (0, 0), (1, 1200), (1200, 1)):
        seg = mini * np.array([w, h, w, h], dtype=float)
        run(f"B/size/{w}x{h}", up.solve_ex, seg, mode="auto", w=w, h=h, focal_px=900.0)
    # detector on matching real images
    for w, h in ((1, 1), (2, 2), (8, 8), (10000, 3)):
        img = np.zeros((h, w), np.uint8)
        if h > 4 and w > 4:
            img[:, w // 2] = 255
            img[h // 2, :] = 255
        run(f"B/detect/{w}x{h}", up.detect_segments, img)


# ----------------------------------------------------------------- C focal
def c_focal():
    for f in (0.0, -1.0, -900.0, float("nan"), float("inf"), float("-inf"),
              1e9, 1e12, 1e-3, 1e-9, 0.5, 1.0, 1e6):
        for mode in ("auto", "guided"):
            seg = GUIDED4 if mode == "guided" else BASE_SEG
            run(f"C/focal/{f!r}/{mode}", up.solve_ex, seg, mode=mode, w=W, h=H, focal_px=f)
    run("C/focal/None", up.solve_ex, BASE_SEG, mode="auto", **{k: v for k, v in S().items() if k != "focal_px"})
    run("C/focal/np.float32", up.solve_ex, BASE_SEG, mode="auto", w=W, h=H, focal_px=np.float32(900))


# ------------------------------------------------------------------ D modes
def d_modes():
    for mode in ("bogus", None, "", "AUTO", "Auto", " auto", 0, ["auto"]):
        run(f"D/mode/{mode!r}", up.solve_ex, BASE_SEG, mode=mode, **S())
    for n in range(0, 9):
        seg = GUIDED4 if n == 4 else (np.tile(GUIDED4, (2, 1))[:n] if n else np.zeros((0, 4)))
        if n > 4:
            seg = np.vstack([GUIDED4, np.tile(GUIDED4, (2, 1))[:n - 4]])
        run(f"D/guided/n={n}", up.solve_ex, seg, mode="guided", **S())
    # guided degenerate quartets
    deg = {
        "all_identical": np.tile([500., 500., 500., 500.], (4, 1)),
        "all_vertical_collinear": np.array([[800., 100., 800., 1100.]] * 4),
        "all_horizontal_collinear": np.array([[100., 400., 1500., 400.]] * 4),
        "two_pairs": np.array([[400., 300., 1200., 900.]] * 4),
        "nan": np.where(np.arange(16).reshape(4, 4) == 5, np.nan, GUIDED4),
        "inf": np.where(np.arange(16).reshape(4, 4) == 5, np.inf, GUIDED4),
        "huge": GUIDED4 * 1e6,
        "zero_len_mixed": np.array([[800., 400., 800., 400.]] * 2 + [[400., 700., 1200., 700.]] * 2),
    }
    for k, seg in deg.items():
        run(f"D/guided/{k}", up.solve_ex, seg, mode="guided", **S())
    # damping override
    for d in (0.0, 0.5, 1.0, 2.0, 100.0, -1.0, (0., 0., 0.), (1., 1., 1.),
              (float("nan"),) * 3, 1e6):
        run(f"D/damping/{d!r}", up.solve_ex, BASE_SEG, mode="full", damping=d, **S())
    run("D/damping/bad_len", up.solve_ex, BASE_SEG, mode="full", damping=(1., 2.), **S())
    run("D/shared_k", up.solve_ex, BASE_SEG, mode="full", shared_k=True, **S())


# -------------------------------------------------------- E framing helpers
def e_framing():
    I = np.eye(3)
    HORIZON_STRADDLE = np.array([[1., 0., 0.], [0., 1., 0.], [0., 1., -0.5]])
    HORIZON_CORNER = np.array([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.]])
    SINGULAR = np.array([[1., 0., 0.], [2., 0., 0.], [0., 1., 1.]])
    SING_SINGULAR_H22_0 = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
    H22_ZERO = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    HUGE_T = np.array([[1., 0., 1e6], [0., 1., -1e6], [0., 0., 1.]])
    NEG_ID = -np.eye(3)
    MIRROR = np.diag([-1., 1., 1.])
    MIRROR_Y = np.diag([1., -1., 1.])
    TINY = np.eye(3) * 1e-320
    BOWTIE = np.array([[1., 0., 0.], [0., 1., 0.], [1., -1., 0.5]])
    ZOOM = np.diag([1e6, 1e6, 1.])
    ROT = up._rotation(0.4, 0.3, 0.2)
    cases = dict(identity=I, horizon_straddle=HORIZON_STRADDLE,
                 horizon_through_corner=HORIZON_CORNER, singular=SINGULAR,
                 singular_h22_zero=SING_SINGULAR_H22_0, h22_zero=H22_ZERO,
                 huge_translation=HUGE_T, neg_identity=NEG_ID, mirror_x=MIRROR,
                 mirror_y=MIRROR_Y, underflow_1e320=TINY, bowtie=BOWTIE,
                 huge_zoom=ZOOM, pure_rotation=ROT,
                 nan=np.where(np.arange(9).reshape(3, 3) == 4, np.nan, I),
                 inf=np.where(np.arange(9).reshape(3, 3) == 4, np.inf, I),
                 wrong_shape=np.eye(4), nonfinite_H22=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., np.nan]]))
    for k, Hm in cases.items():
        run(f"E/frame_output/{k}", up.frame_output, Hm)
        run(f"E/scale_output/{k}", up.scale_output, Hm)
    # full pipeline with framing/scaling on the real solver
    for mode in ("auto", "level", "full", "vertical", "guided"):
        seg = GUIDED4 if mode == "guided" else BASE_SEG
        run(f"E/pipeline/framing/{mode}", up.solve_ex, seg, mode=mode, framing=True, **S())
        run(f"E/pipeline/scaling/{mode}", up.solve_ex, seg, mode=mode, framing=True, scaling=True, **S())
    run("E/pipeline/framing_and_scaling_auto", up.solve_ex, BASE_SEG, mode="auto",
        framing=True, scaling=True, **S())


# ---------------------------------------------------------------- F detector
def f_detector():
    rng = np.random.default_rng(7)
    flat = np.full((64, 64), 128, np.uint8)
    noise = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    one = np.array([[200]], np.uint8)
    big = rng.integers(0, 256, (2160, 3840), dtype=np.uint8)
    blank4k = np.full((2160, 3840), 128, np.uint8)
    striped4k = np.zeros((2160, 3840), np.uint8)
    striped4k[:, ::40] = 255
    run("F/detect/flat_grey_64", up.detect_segments, flat)
    run("F/detect/noise_64", up.detect_segments, noise)
    run("F/detect/one_pixel", up.detect_segments, one)
    run("F/detect/noise_4k_default", up.detect_segments, big)
    run("F/detect/flat_4k_default", up.detect_segments, blank4k)
    run("F/detect/striped_4k_default", up.detect_segments, striped4k)
    run("F/detect/noise_4k_fullres", up.detect_segments, big, max_side=4000)
    run("F/detect/noise_4k_max_side_1e9", up.detect_segments, big, max_side=10**9)
    run("F/detect/flat_max_side_0", up.detect_segments, flat, max_side=0)
    run("F/detect/nan_image", up.detect_segments, np.array([[np.nan, 1.], [1., 1.]]))
    run("F/detect/empty_image", up.detect_segments, np.zeros((0, 0)))
    run("F/detect/float_flat", up.detect_segments, np.full((32, 32), 0.5))
    run("F/solve_from_image_noise", up.solve_ex, image=noise, mode="auto")
    run("F/solve_from_image_flat", up.solve_ex, image=flat, mode="auto")
    run("F/solve_from_image_1px", up.solve_ex, image=one, mode="auto")


if __name__ == "__main__":
    for grp in (a_segments, b_sizes, c_focal, d_modes, e_framing, f_detector):
        print(f"\n=== {grp.__name__} ===")
        grp()
    with open("/Users/rohan/Projects/trichroma/research/upright/evidence/port_fuzz_raw.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    print("\nwrote port_fuzz_raw.json")
