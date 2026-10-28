"""Second fuzz pass: invariants, natural silent-bad search, memory, extras."""
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
ALARM = 120
OUT = []


class Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(Timeout("alarm")))


def scene(p, y, r, f, w=W, h=H):
    R = up._rotation(math.radians(p), math.radians(y), math.radians(r))
    Rt = R.T
    out = []

    def proj(q):
        q = Rt @ np.asarray(q, float)
        if q[2] <= 0.5:
            return None
        return np.array([f * q[0] / q[2] + w / 2.0, f * q[1] / q[2] + h / 2.0])

    for X in (-3, -1.5, 0, 1.5, 3):
        for Z in (5.0, 7.0):
            a, b = proj([X, -2.0, Z]), proj([X, 3.0, Z])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                out.append([*a, *b])
    for Y in (-1.5, 0.0, 1.5):
        for Z in (5.0, 7.0):
            a, b = proj([-4.0, Y, Z]), proj([4.0, Y, Z])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                out.append([*a, *b])
    for Y in (-1.0, 1.0):
        for X in (-2.0, 2.0):
            a, b = proj([X, Y, 4.0]), proj([X, Y, 9.0])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                out.append([*a, *b])
    return np.asarray(out, float)


BASE = scene(15, 6, 3, 900.0)
BASE_H = np.array([[700., 200., 700., 1000.], [900., 250., 900., 1050.],
                   [200., 400., 1400., 420.], [150., 800., 1450., 780.]])


def assess(H):
    bad = []
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        return ["shape"], {}
    if not np.all(np.isfinite(H)):
        bad.append("non-finite H")
    if H[2, 2] == 0 or not np.isfinite(H[2, 2]):
        bad.append("H22 zero/non-finite")
    if np.all(np.isfinite(H)):
        s = np.linalg.svd(H, compute_uv=False)
        if not s[-1] > 1e-12 * s[0]:
            bad.append("singular/rank-deficient")
        elif s[0] / s[-1] > 1e12:
            bad.append("cond %.3g" % (s[0] / s[-1]))
    C = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
    P = C @ H.T
    z = P[:, 2]
    if not np.all(np.isfinite(z)):
        bad.append("non-finite corner w")
    elif np.any(z == 0):
        bad.append("corner on horizon")
    elif not (np.all(z > 0) or np.all(z < 0)):
        bad.append("corners straddle horizon")
    elif np.all(np.isfinite(P)):
        xy = P[:, :2] / P[:, 2:3]
        area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                         - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
        if area < 1e-12:
            bad.append("degenerate quad")
        if np.max(np.abs(xy)) > 1e6:
            bad.append("magnitude %.3g" % np.max(np.abs(xy)))
        return bad, dict(area=float(area), cond=float(np.linalg.cond(H)))
    return bad, {}


def rec(case, verdict, note="", **kw):
    r = dict(case=case, verdict=verdict, note=note, **kw)
    OUT.append(r)
    print(f"{verdict:14s} {case}  {note}")


def run(case, fn, *a, **kw):
    t0 = time.perf_counter()
    signal.alarm(ALARM)
    try:
        out = fn(*a, **kw)
    except Timeout:
        signal.alarm(0)
        rec(case, "HANG", ">120s")
        return None
    except Exception as e:  # noqa: BLE001
        signal.alarm(0)
        rec(case, "RAISE", f"{type(e).__name__}: {e}")
        return None
    signal.alarm(0)
    ms = (time.perf_counter() - t0) * 1000
    if isinstance(out, dict):
        ok = bool(out.get("ok"))
        if ok:
            bad, st = assess(out["H"])
            rec(case, "SILENT_BAD" if bad else "ok", str(bad) if bad else "",
                ms=round(ms, 1), reason=out.get("reason"))
        else:
            rec(case, "fail", str(out.get("reason"))[:120], ms=round(ms, 1))
        return out
    if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] == 4:
        rec(case, "segments", f"n={len(out)}", ms=round(ms, 1))
        return out
    bad, st = assess(out)
    rec(case, "SILENT_BAD" if bad else "ok", str(bad) if bad else "", ms=round(ms, 1))
    return out


# ---------------------------------------------------------- invariants
def invariants():
    print("\n== invariants ==")
    I = np.eye(3)
    R = up._rotation(0.35, 0.25, 0.15)
    big_t = np.array([[1., 0., 1e6], [0., 1., -1e6], [0., 0., 1.]])
    for name, Hm in (("identity", I), ("rotation", R), ("big_translation", big_t),
                     ("framed_big_t", up.frame_output(big_t))):
        F = up.frame_output(Hm)
        # commutation: scale_output(H) must equal scale_output(frame_output(H))
        S1 = up.scale_output(Hm)
        S2 = up.scale_output(F)
        d = float(np.max(np.abs(S1 / S1[2, 2] - S2 / S2[2, 2])))
        rec(f"inv/commute/{name}", "BAD" if d > 1e-9 else "ok", f"max diff {d:.3g}")
        # frame_output must be idempotent in translation
        FF = up.frame_output(F)
        d2 = float(np.max(np.abs(F / F[2, 2] - FF / FF[2, 2])))
        rec(f"inv/idempotent/{name}", "BAD" if d2 > 1e-9 else "ok", f"max diff {d2:.3g}")
        # homogeneous sign invariance
        for s in (-1.0, 2.0, 1e-6):
            try:
                G = up.frame_output(s * Hm)
                dd = float(np.max(np.abs(G / G[2, 2] - F / F[2, 2])))
                rec(f"inv/scale_invariance/{name}/{s:g}", "BAD" if dd > 1e-7 else "ok",
                    f"max diff {dd:.3g}")
            except Exception as e:  # noqa: BLE001
                rec(f"inv/scale_invariance/{name}/{s:g}", "RAISE",
                    f"{type(e).__name__}: {e}")
    # corners same-side check for every ok=True pipeline output
    for mode in ("auto", "level", "full", "vertical", "guided"):
        seg = BASE_H if mode == "guided" else BASE
        o = up.solve_ex(seg, w=W, h=H, focal_px=900.0, mode=mode)
        if o["ok"]:
            P = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]]) @ o["H"].T
            rec(f"inv/corners/{mode}", "ok" if np.all(P[:, 2] > 0) else "BAD",
                f"z={np.round(P[:,2],4).tolist()}")
    # framing rescues a horizon-crossing H
    o = up.solve_ex(BASE, w=W, h=H, focal_px=900.0, mode="full", damping=100.0)
    rec("inv/damping100_unframed", "SILENT_BAD" if o["ok"] else "fail", str(o["reason"]))
    o = up.solve_ex(BASE, w=W, h=H, focal_px=900.0, mode="full", damping=100.0,
                    framing=True)
    rec("inv/damping100_framed", "SILENT_BAD" if o["ok"] else "fail", str(o["reason"]))
    # scaling without framing
    o = up.solve_ex(BASE, w=W, h=H, focal_px=900.0, mode="auto", scaling=True)
    rec("inv/scaling_without_framing", "SILENT_BAD" if o["ok"] else "fail",
        str(o["reason"]))


# --------------------------------------------------- extras / crash probes
def extras():
    print("\n== extras ==")
    # 1x1 with real structure -> max(w,h)-1 == 0
    s1 = BASE / np.array([W, H, W, H])  # all coords in [0,1]
    run("x/1x1_structure", up.solve_ex, s1, w=1, h=1, focal_px=900.0, mode="auto")
    run("x/1x1_guided", up.solve_ex, BASE_H / np.array([W, H, W, H]), w=1, h=1,
        focal_px=900.0, mode="guided")
    run("x/1x1_off", up.solve_ex, s1, w=1, h=1, focal_px=900.0, mode="off")
    # underflow after normalisation
    run("x/frame_underflow", up.frame_output, np.diag([1., 1., 1e-320]))
    run("x/scale_underflow", up.scale_output, np.diag([1., 1., 1e-320]))
    # negative / zero / float sizes
    for w, h in ((-1600, -1200), (1600.0, 1200.0), (0, 1200), (1600, 0), (1, 1)):
        run(f"x/size/{w}x{h}", up.solve_ex, BASE, w=w, h=h, focal_px=900.0, mode="auto")
    # focal passed as a string / array
    run("x/focal_str", up.solve_ex, BASE, w=W, h=H, focal_px="900", mode="auto")
    run("x/focal_list", up.solve_ex, BASE, w=W, h=H, focal_px=[900.0], mode="auto")
    run("x/focal_bool", up.solve_ex, BASE, w=W, h=H, focal_px=True, mode="auto")
    # image dtypes / channel counts
    rng = np.random.default_rng(3)
    g = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    run("x/img/bgr", up.detect_segments, np.dstack([g, g, g]))
    run("x/img/rgba", up.detect_segments, np.dstack([g, g, g, g]))
    run("x/img/bool", up.detect_segments, g > 128)
    run("x/img/uint16", up.detect_segments, g.astype(np.uint16) * 200)
    run("x/img/float64", up.detect_segments, g.astype(np.float64) / 255)
    run("x/img/list", up.detect_segments, g.tolist())
    # huge N
    rng = np.random.default_rng(11)
    many = rng.uniform(0, 1600, (100000, 4))
    many[:, 2] = many[:, 0] + rng.uniform(20, 200, 100000)
    many[:, 3] = many[:, 1] + rng.uniform(20, 200, 100000)
    run("x/many_segments_100k", up.solve_ex, many, w=W, h=H, focal_px=900.0, mode="auto")
    # mode off still runs the detector
    big = rng.integers(0, 256, (2160, 3840), dtype=np.uint8)
    run("x/off_with_4k_image", up.solve_ex, image=big, mode="off")
    run("x/off_with_4k_image_max_side_huge", up.solve_ex, image=big, mode="off",
        max_side=10**9)


# ------------------------------------------------ natural silent-bad search
def hunt(n=400):
    print("\n== natural silent-bad hunt ==")
    rng = np.random.default_rng(1234)
    seen = {}
    stats = dict(total=0, ok=0, bad=0, fail=0, raise_=0)
    for i in range(n):
        p = rng.uniform(0, 40)
        y = rng.uniform(-30, 30)
        r = rng.uniform(-15, 15)
        f = float(rng.choice([200, 350, 500, 700, 900, 1200, 2000, 5000, 20000, 1e6]))
        mode = str(rng.choice(["auto", "level", "full", "vertical", "guided"]))
        seg = scene(p, y, r, f)
        if len(seg) < 4:
            continue
        stats["total"] += 1
        if mode == "guided":
            if len(seg) < 4:
                continue
            seg = seg[:4]
            if len(np.unique(np.where(np.abs(seg[:, 3] - seg[:, 1]) >=
                                      np.abs(seg[:, 2] - seg[:, 0]), 0, 1))) < 2:
                continue
        try:
            o = up.solve_ex(seg, w=W, h=H, focal_px=f, mode=mode)
        except Exception as e:  # noqa: BLE001
            stats["raise_"] += 1
            key = "RAISE:" + type(e).__name__
            if key not in seen:
                seen[key] = dict(p=p, y=y, r=r, f=f, mode=mode, note=f"{type(e).__name__}: {e}")
            continue
        if not o["ok"]:
            stats["fail"] += 1
            continue
        stats["ok"] += 1
        bad, st = assess(o["H"])
        if bad:
            stats["bad"] += 1
            key = ",".join(sorted(bad))
            if key not in seen:
                seen[key] = dict(p=p, y=y, r=r, f=f, mode=mode, note=key,
                                 cond=st.get("cond"), area=st.get("area"))
    for k, v in seen.items():
        print("  signature:", k, v)
        d = dict(case="hunt", verdict="SILENT_BAD")
        d.update(v)
        OUT.append(d)
    print("  stats", stats)
    OUT.append(dict(case="hunt/stats", verdict="info", note=json.dumps(stats)))


# ------------------------------------------- guided degenerate quartets
def guided_deg(n=300):
    print("\n== guided near-degenerate ==")
    rng = np.random.default_rng(99)
    seen = {}
    for i in range(n):
        kind = i % 5
        if kind == 0:  # near-parallel verticals + horizontals
            v = 700 + rng.uniform(-3, 3)
            seg = np.array([[v, 200, v + 0.5, 1000], [v + 200, 200, v + 200, 1000],
                            [200, 400, 1400, 400], [200, 800, 1400, 800]])
        elif kind == 1:  # two verticals only, near-identical
            seg = np.tile([700., 200., 700., 1000.], (4, 1)) + rng.normal(0, 1, (4, 4))
        elif kind == 2:  # all horizontals, tiny slope differences
            seg = np.tile([200., 400., 1400., 400.], (4, 1)) + rng.normal(0, 2, (4, 4))
        elif kind == 3:  # crossing / near-duplicate pairs
            seg = np.array([[700., 200., 700., 1000.], [700.0001, 200., 700.0001, 1000.],
                            [200, 400, 1400, 400], [200.0001, 400, 1400, 400]])
        else:  # one long + three tiny
            seg = np.array([[700., 200., 700., 1000.], [700., 500., 701., 500.],
                            [700., 501., 701., 501.], [700., 502., 701., 502.]])
        try:
            o = up.solve_ex(seg, w=W, h=H, focal_px=900.0, mode="guided")
        except Exception as e:  # noqa: BLE001
            key = "RAISE:" + type(e).__name__ + ":" + str(e)[:40]
            if key not in seen:
                seen[key] = dict(kind=kind, seg=seg.tolist())
            continue
        if o["ok"]:
            bad, st = assess(o["H"])
            if bad:
                key = "BAD:" + ",".join(sorted(bad))
                if key not in seen:
                    seen[key] = dict(kind=kind, seg=seg.tolist(),
                                     resid=o["guide_residual_deg"])
    for k, v in seen.items():
        print("  ", k, json.dumps(v)[:300])
        OUT.append(dict(case="guided_deg", verdict="SILENT_BAD", note=k, detail=v))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "inv"):
        invariants()
    if which in ("all", "extras"):
        extras()
    if which in ("all", "hunt"):
        hunt()
    if which in ("all", "guided"):
        guided_deg()
    with open("/Users/rohan/Projects/trichroma/research/upright/evidence/port_fuzz_raw2.json", "w") as fh:
        json.dump(OUT, fh, indent=1)
