"""Corpus measurement of Adobe Auto's damping from its own stored transforms.

Read-only on the catalog. Reuses only pure geometry helpers from the port.

Per photo (report section 3.1 two-camera model): RQ(P H K0) = K1, R with K0 fixed to
the stored SOURCE camera. Full is the rectification reference, so the input pose is
R0 = R_full^T and the input convergence is DEFINED as the convergence Full removes
exactly: c_in = conv(R0, K0). Auto's residual is conv(R_auto R0, K1_auto).
removed = 1 - residual / input. Cross-checks: Vertical slot (4) must imply the same
pitch; applied-rotation ratio must reproduce the report's published 3.5 table.
"""
import json, math, sys
import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares

sys.path.insert(0, "port")
import upright_port as up

SRC = "/tmp/auto_corpus_slots.jsonl"
GRID = (61, 41)
EDGES = [0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 27, 35, 46]


def grid():
    xs = (np.arange(GRID[0]) + 0.5) / GRID[0]
    ys = (np.arange(GRID[1]) + 0.5) / GRID[1]
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])


def split(Hn, K0, w, h):
    P = np.diag([w, h, 1.0])
    K1, R = rq(P @ Hn @ K0)
    signs = np.diag(np.where(np.diag(K1) < 0, -1.0, 1.0))
    K1, R = K1 @ signs, signs @ R
    if np.linalg.det(R) < 0:
        return None
    err = float(np.max(np.abs(np.linalg.inv(P) @ K1 @ R @ np.linalg.inv(K0) - Hn)))
    return np.linalg.inv(P) @ K1, R, err


def conv(R, K, w, h, g):
    vp = K @ R @ np.array([0.0, 1.0, 0.0])
    line = np.cross(g, vp)
    n = np.hypot(line[:, 0], line[:, 1])
    n[n < 1e-15] = 1e-15
    line = line / n[:, None]
    tilt = np.arctan2(-line[:, 1] * w, line[:, 0] * h)
    tilt = (tilt + math.pi / 2) % math.pi - math.pi / 2
    return math.degrees(float(np.diff(np.percentile(tilt, [5, 95]))[0]))


def port_curve(t_deg):
    lam = up.L0_V * math.exp(-(math.radians(t_deg) ** 2) / (2.0 * up.SIGMA_V ** 2))
    return lam / (lam + 1.0)


g = grid()


def measure(centre):
    rows, bad = [], dict(rq=0, rebuild=0, zero=0, nonfinite=0)
    for line in open(SRC):
        d = json.loads(line)
        w, h = float(d["width"]), float(d["height"])
        m = d["meta"]
        f35 = m.get("UprightFocalLength35mm")
        if not f35 or not (5.0 <= f35 <= 500.0):
            continue
        K0 = up._intrinsics_norm(up.focal_px_from_35mm(f35, w, h), w, h)
        if centre:
            K0[0, 2] = m.get("UprightCenterNormX", 0.5)
            K0[1, 2] = m.get("UprightCenterNormY", 0.5)
        tr, Hs = d["transform"], {}
        for slot in ("1", "2", "4"):
            if slot not in tr:
                continue
            H = np.array(tr[slot], dtype=float).reshape(3, 3)
            if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
                Hs = None
                break
            Hs[slot] = H / H[2, 2]
        if not Hs or "1" not in Hs or "2" not in Hs:
            continue
        a, f = split(Hs["1"], K0, w, h), split(Hs["2"], K0, w, h)
        if a is None or f is None:
            bad["rq"] += 1
            continue
        K1a, Ra, ea = a
        K1f, Rf, ef = f
        if max(ea, ef) > 1e-9:
            bad["rebuild"] += 1
            continue
        R0 = Rf.T
        c_in, c_auto = conv(R0, K0, w, h, g), conv(Ra @ R0, K1a, w, h, g)
        if not (np.isfinite(c_in) and np.isfinite(c_auto)):
            bad["nonfinite"] += 1
            continue
        if c_in < 1e-6:
            bad["zero"] += 1
            continue
        pitch_in, yaw_in, roll_in = np.degrees(up._euler_xyz(R0))
        pitch_v = None
        if "4" in Hs:
            v = split(Hs["4"], K0, w, h)
            if v is not None and v[2] < 1e-9:
                pitch_v = float(np.degrees(up._euler_xyz(v[1].T))[0])
        rows.append(dict(id=d["image_id"], c_in=c_in, c_auto=c_auto,
                         removed=1.0 - c_auto / c_in,
                         pitch_in=float(pitch_in), yaw_in=float(yaw_in),
                         roll_in=float(roll_in), pitch_v=pitch_v,
                         rot_full=float(np.degrees(np.arccos(
                             np.clip((np.trace(Rf) - 1) / 2, -1, 1)))),
                         rot_auto=float(np.degrees(np.arccos(
                             np.clip((np.trace(Ra) - 1) / 2, -1, 1))))))
    return rows, bad


def select(rows):
    k = [r for r in rows
         if 1.5 <= r["c_in"] <= 45.0 and abs(r["roll_in"]) <= 12.0
         and -0.6 <= r["removed"] <= 1.3 and abs(r["pitch_in"]) <= 45.0
         and abs(r["yaw_in"]) <= 45.0]
    c = [r for r in k if r["pitch_v"] is not None
         and abs(r["pitch_v"] - r["pitch_in"]) <= 6.0]
    return k, c


rows_c, bad = measure(True)
print(f"decomposed={len(rows_c)} rq_fail={bad['rq']} rebuild_fail={bad['rebuild']} "
      f"zero={bad['zero']} nonfinite={bad['nonfinite']}", file=sys.stderr)
keep_all, keep = select(rows_c)
print(f"filtered={len(keep_all)} crosscheck_ok={len(keep)}", file=sys.stderr)
rows_n, _ = measure(False)
_, keep_nc = select(rows_n)
print(f"centred_K0_crosscheck_ok={len(keep_nc)}", file=sys.stderr)


def table(rs, key, minn=8):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = [r for r in rs if lo <= abs(r[key]) < hi]
        if len(s) < minn:
            out.append((lo, hi, len(s), None))
            continue
        v = np.array([r["removed"] for r in s])
        out.append((lo, hi, len(s), (float(np.median(v)), float(v.mean()),
                                     float(np.percentile(v, 25)),
                                     float(np.percentile(v, 75)))))
    return out


def fmt(tb, label, pc=False):
    head = f"{'bin(deg)':>10} {'n':>5} {'median':>8} {'mean':>8} {'p25':>7} {'p75':>7}"
    lines = [f"\n[{label}]", head + (f" {'port':>7}" if pc else "")]
    for lo, hi, n, st in tb:
        if st is None:
            lines.append(f"{lo:4d}-{hi:<5d} {n:5d} {'-':>8} {'-':>8} {'-':>7} {'-':>7}")
            continue
        med, mean, p25, p75 = st
        tail = f" {port_curve(0.5 * (lo + hi)):7.3f}" if pc else ""
        lines.append(f"{lo:4d}-{hi:<5d} {n:5d} {med:8.3f} {mean:8.3f} {p25:7.3f} {p75:7.3f}{tail}")
    return "\n".join(lines)


# stratified sample: floor of 40 per occupied bin, remaining mass random
rng = np.random.default_rng(20261028)
strat = []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    s = [r for r in keep if lo <= abs(r["pitch_in"]) < hi]
    if not s:
        continue
    take = min(len(s), 40)
    strat += [s[i] for i in rng.choice(len(s), take, replace=False)]
rand = [keep[i] for i in rng.choice(len(keep), 500, replace=False)]
print(f"strat_n={len(strat)} rand_n={len(rand)}", file=sys.stderr)

print("=" * 78)
print("CORPUS: Adobe Auto damping, measured from 12,075 decomposed native transforms")
print("=" * 78)
print(f"catalog photos carrying non-identity Auto AND Full: 13287")
print(f"selected (1.5<=c_in<=45, |roll|<=12, |pitch|,|yaw|<=45, 0<=removed<=1): {len(keep_all)}")
print(f"of those, Full-derived pitch agrees with Vertical-derived pitch (<=6deg): {len(keep)}")
print(f"primary set = {len(keep)}; random 500 = {len(rand)}; stratified = {len(strat)}")
print(f"median |pitch| of primary set: {np.median([abs(r['pitch_in']) for r in keep]):.3f} deg")
print(f"median c_in: {np.median([r['c_in'] for r in keep]):.3f} deg")

print(fmt(table(strat, "pitch_in"), "removed fraction vs |input pitch|, STRATIFIED (40/bin floor)", True))
print(fmt(table(rand, "pitch_in"), "removed fraction vs |input pitch|, RANDOM 500", True))
print(fmt(table(keep, "pitch_in"), f"removed fraction vs |input pitch|, ALL n={len(keep)}", True))
print(fmt(table(strat, "c_in"), "removed fraction vs input vertical convergence (keystone units)", True))
print(fmt(table(strat, "yaw_in"), "removed fraction vs |input yaw| (control: flat => pitch is the axis)"))

# bimodality: fraction of photos where Auto is near a no-op or near full
print("\n[branch] share of photos at the extremes, by |input pitch| (random 500)")
print(f"  {'bin':>10} {'n':>5} {'removed<0.15':>13} {'removed>0.85':>13} {'median':>8}")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    s = [r["removed"] for r in rand if lo <= abs(r["pitch_in"]) < hi]
    if len(s) < 8:
        continue
    v = np.array(s)
    print(f"  {lo:4d}-{hi:<5d} {len(s):5d} {np.mean(v < 0.15)*100:12.1f}% "
          f"{np.mean(v > 0.85)*100:12.1f}% {np.median(v):8.3f}")

# fine low-tilt bins on the full primary set
print("\n[fine] removed fraction at low |input pitch| (all n=4966)")
for lo, hi in ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 5)):
    v = np.array([r["removed"] for r in keep if lo <= abs(r["pitch_in"]) < hi])
    if len(v) < 8:
        print(f"  {lo:4.1f}-{hi:<4.1f} {v.size:5d} {'-':>8}")
        continue
    print(f"  {lo:4.1f}-{hi:<4.1f} {v.size:5d} med={np.median(v):6.3f} "
          f"mean={v.mean():6.3f} p25={np.percentile(v,25):6.3f} p75={np.percentile(v,75):6.3f} "
          f"n_lt15={np.mean(v<0.15)*100:5.1f}%")
print("  (c_in floor is 1.5 deg, so pitch<1 is under-represented)")

# cross-check 1: applied-rotation ratio must reproduce report 3.5's published table
print("\n[cross-check A] applied-rotation ratio rot_auto/rot_full vs |rot_full|")
print("  report 3.5 published: 2-3:0.744 3-4:0.622 4-5:0.548 5-6:0.461 6-7:0.373 "
      "7-8:0.311 8-10:0.201 10-12:0.118 12-14:0.090 14-17:0.062 17-20:0.043 20-25:0.020 25-30:0.012")
rs = [r for r in keep if r["rot_full"] > 0.05]
print(f"  {'bin':>10} {'n':>5} {'median ratio':>12}   report")
rep = {(2, 3): .744, (3, 4): .622, (4, 5): .548, (5, 6): .461, (6, 7): .373,
       (7, 8): .311, (8, 10): .201, (10, 12): .118, (12, 14): .090,
       (14, 17): .062, (17, 20): .043, (20, 25): .020, (25, 30): .012}
for lo, hi, rv in ([(0, 1, None), (1, 2, None)] + [(a, b, v) for (a, b), v in rep.items()]):
    s = [r for r in rs if lo <= r["rot_full"] < hi]
    tag = f"{rv:.3f}" if rv is not None else "(not published)"
    if len(s) < 8:
        print(f"  {lo:4d}-{hi:<5d} {len(s):5d} {'-':>12}   {tag}")
        continue
    q = np.median([min(r["rot_auto"] / r["rot_full"], 5.0) for r in s])
    print(f"  {lo:4d}-{hi:<5d} {len(s):5d} {q:12.3f}   {tag}")

# cross-check 2: gauge sensitivity
print("\n[cross-check B] gauge sensitivity, centred K0 vs stored principal point")
print(f"  {'bin':>10} {'stored':>8} {'centred':>8} {'delta':>8}")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    a = [r["removed"] for r in keep if lo <= abs(r["pitch_in"]) < hi]
    b = [r["removed"] for r in keep_nc if lo <= abs(r["pitch_in"]) < hi]
    if len(a) < 8 or len(b) < 8:
        continue
    ma, mb = np.median(a), np.median(b)
    print(f"  {lo:4d}-{hi:<5d} {ma:8.3f} {mb:8.3f} {mb - ma:+8.3f}")

# 2D pitch x yaw: is pitch the axis, or yaw?
print("\n[cross-check C] median removed, pitch x yaw grid (n>=15)")
print(f"  {'pitch\\yaw':>10} " + " ".join(f"{a:>9}" for a in ("0-5", "5-15", "15-45")))
for lo, hi in ((0, 4), (4, 10), (10, 18), (18, 30), (30, 46)):
    cells = []
    for ylo, yhi in ((0, 5), (5, 15), (15, 45)):
        s = [r["removed"] for r in keep
             if lo <= abs(r["pitch_in"]) < hi and ylo <= abs(r["yaw_in"]) < yhi]
        cells.append(f"{np.median(s):9.3f}({len(s)})" if len(s) >= 15 else f"{'-':>9}({len(s)})")
    print(f"  {lo:4d}-{hi:<5d}  " + " ".join(cells))

# fits on stratified pitch-bin medians
tb = [(lo, hi, n, st) for lo, hi, n, st in table(strat, "pitch_in", 20) if st]
x = np.array([0.5 * (lo + hi) for lo, hi, n, st in tb])
y = np.array([st[0] for lo, hi, n, st in tb])
wt = np.array([n for lo, hi, n, st in tb], dtype=float)
print(f"\n[fit] x={np.round(x, 2).tolist()}")
print(f"[fit] measured bin medians y={np.round(y, 4).tolist()}")
print(f"[fit] port curve at x={np.round([port_curve(v) for v in x], 4).tolist()}")
print(f"[fit] port minus measured={np.round([port_curve(v) for v in x] - y, 4).tolist()}")

forms = {
    "monotone lam/(lam+1) free L0,sigma (port's form)": (
        lambda p: (p[0] * np.exp(-x ** 2 / (2 * p[1] ** 2))) /
                  (p[0] * np.exp(-x ** 2 / (2 * p[1] ** 2)) + 1), [2.64, 13.33]),
    "gaussian a*exp(-x^2/2s^2)+c": (
        lambda p: p[0] * np.exp(-x ** 2 / (2 * p[1] ** 2)) + p[2], [1.0, 8.0, 0.0]),
    "hump: a*exp(-x^2/2s^2)*1/(1+(x/t1)^4)+c": (
        lambda p: p[0] * np.exp(-x ** 2 / (2 * p[1] ** 2)) / (1 + (x / p[2]) ** 4) + p[3],
        [1.0, 12.0, 20.0, 0.0]),
    "lognormal hump a*exp(-(ln x-mu)^2/2s^2)": (
        lambda p: p[0] * np.exp(-(np.log(np.maximum(x, 1e-3)) - p[1]) ** 2 / (2 * p[2] ** 2)),
        [0.9, 2.2, 0.7]),
}
for name, (fn, p0) in forms.items():
    lo = [1e-3, 0.05, -1.0, 0.0][:len(p0)]
    hi = [1e3, 200.0, 200.0, 1.0][:len(p0)]
    p0 = [min(max(v, lo[i] + 1e-6), hi[i] - 1e-6) for i, v in enumerate(p0)]
    res = least_squares(lambda p: np.sqrt(wt) * (fn(p) - y), p0, bounds=(lo, hi),
                        max_nfev=40000)
    pred = fn(res.x)
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    aic = x.size * math.log(max(rms ** 2, 1e-12)) + 2 * len(res.x)
    print(f"  {name:<46} p={[round(float(v), 4) for v in res.x]} rms={rms:.4f} AIC={aic:.1f}")
    print(f"      pred={np.round(pred, 4).tolist()}")

# grid-vs-detected support offset: report 3.5b gives 15.5 deg detected for a 22.8 deg
# true pitch where this grid gives 18.691, so the port's own argument is ~c_in/1.206
OFF = 18.691 / 15.5
print(f"\n[offset] grid/detected convergence on the report's 22.8deg scene = {OFF:.4f}; "
      f"port curve re-read at c_in/{OFF:.4f}")
print(f"  {'bin':>10} {'measured':>9} {'port':>8} {'port_adj':>9} {'d_adj':>8}")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    a = [r["removed"] for r in rand if lo <= abs(r["pitch_in"]) < hi]
    if len(a) < 8:
        continue
    m0 = float(np.median(a))
    pa = port_curve(0.5 * (lo + hi) / OFF)
    print(f"  {lo:4d}-{hi:<5d} {m0:9.3f} {port_curve(0.5*(lo+hi)):8.3f} {pa:9.3f} {pa-m0:+8.3f}")
d_adj = np.array([port_curve(r["c_in"] / OFF) - r["removed"] for r in rand])
print(f"  per-photo: median={np.median(d_adj):+.4f} RMS={np.sqrt(np.mean(d_adj**2)):.4f} "
      f"|d|>0.20 {np.mean(np.abs(d_adj)>0.20)*100:.1f}% port_higher {np.mean(d_adj>0)*100:.1f}%")

print(f"\n[port] L0_V={up.L0_V} SIGMA_V={math.degrees(up.SIGMA_V):.2f}deg "
      f"max_fraction={up.L0_V / (up.L0_V + 1):.6f}")
d_cin = np.array([port_curve(r["c_in"]) - r["removed"] for r in rand])
print(f"[divergence] port - measured over random 500, arg=each photo's own c_in:")
print(f"  median={np.median(d_cin):+.4f} mean={d_cin.mean():+.4f} "
      f"p10={np.percentile(d_cin,10):+.4f} p90={np.percentile(d_cin,90):+.4f}")
print(f"  |d|>0.10: {np.mean(np.abs(d_cin)>0.10)*100:.1f}%   |d|>0.20: "
      f"{np.mean(np.abs(d_cin)>0.20)*100:.1f}%   port over-removes: {np.mean(d_cin>0)*100:.1f}%")
print(f"  RMS={np.sqrt(np.mean(d_cin**2)):.4f}")
json.dump({"n_keep": len(keep), "strat": strat, "rand": rand},
          open("/tmp/auto_corpus_rows.json", "w"))
