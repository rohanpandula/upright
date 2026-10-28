import json, sys
import numpy as np
SENSOR = 36.0
def K_of(f, cx, cy): return np.array([[f,0,cx],[0,f,cy],[0,0,1.0]])
def to_pix(Hn, w, h):
    S = np.diag([1.0/w, 1.0/h, 1.0]); return np.linalg.inv(S) @ Hn @ S
def polar_rot(M):
    U,_,Vt = np.linalg.svd(M); R = U@Vt
    if np.linalg.det(R) < 0: R = U @ np.diag([1,1,-1]) @ Vt
    return R
def decomp(Hpix, f, cx, cy):
    K = K_of(f,cx,cy); R = polar_rot(np.linalg.inv(K) @ Hpix @ K)
    ang = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    ry = np.degrees(np.arcsin(np.clip(R[0,2],-1,1)))
    rx = np.degrees(np.arctan2(-R[1,2], R[2,2]))
    rz = np.degrees(np.arctan2(-R[0,1], R[0,0]))
    return ang, rx, ry, rz
def ident(H): return np.abs(np.array(H).reshape(3,3)-np.eye(3)).max() < 1e-9
def main(path="/tmp/lr_all.jsonl"):
    cols = {k: [] for k in ("img","tilt_full","tilt_auto","rx_full","rx_auto","ry_full","ry_auto",
        "rz_full","rz_auto","rz_level","rz_vert","tilt_level","tilt_vert","rx_vert","ry_vert","rx_level","ry_level")}
    skipped = 0
    for line in open(path):
        try: d = json.loads(line)
        except Exception: continue
        m, tr = d.get("meta") or {}, d.get("transform") or {}
        if any(k not in tr for k in ("1","2","3","4")) or ident(tr["2"]):
            skipped += 1; continue
        w,h = float(d["width"]), float(d["height"])
        f35 = m.get("UprightFocalLength35mm") or 35.0
        f_px = f35/SENSOR*max(w,h)
        cx = (m.get("UprightCenterNormX") or 0.5)*w; cy = (m.get("UprightCenterNormY") or 0.5)*h
        try:
            vals = {}
            for name, slot in (("full","2"),("auto","1"),("level","3"),("vert","4")):
                vals[name] = decomp(to_pix(np.array(tr[slot]).reshape(3,3), w, h), f_px, cx, cy)
        except Exception: skipped += 1; continue
        cols["img"].append(d["image_id"])
        for name in ("full","auto","level","vert"):
            a,rx,ry,rz = vals[name]
            cols["tilt_"+name].append(a); cols["rx_"+name].append(rx)
            cols["ry_"+name].append(ry); cols["rz_"+name].append(rz)
    return {k: np.array(v,dtype=float) for k,v in cols.items()}, skipped
c, skipped = main()
print(f"usable {len(c['img'])}   skipped/abstained {skipped}")
k_full = np.hypot(c["rx_full"], c["ry_full"]); k_auto = np.hypot(c["rx_auto"], c["ry_auto"])
roll = np.abs(c["rz_full"])
m = (k_full > 2.0) & (k_full < 40.0) & (roll < 5.0)
print(f"\nphotos with real keystone (2-40 deg) away from roll: {m.sum()}")
edges = [2,3,4,5,6,7,8,10,12,14,17,20,25,30,40]
print(f"\n{'keystone bin':>13} {'n':>6} {'med ratio':>10} {'mean':>8} {'p25':>7} {'p75':>7}")
xs, ys, ns = [], [], []
for a,b in zip(edges[:-1], edges[1:]):
    s = m & (k_full >= a) & (k_full < b); n = int(s.sum())
    if n < 20: continue
    ratio = np.clip(k_auto[s]/np.maximum(k_full[s],1e-9), 0, 5)
    xs.append(0.5*(a+b)); ys.append(np.median(ratio)); ns.append(n)
    print(f"{a:6.0f}-{b:<6.0f} {n:6d} {np.median(ratio):10.3f} {ratio.mean():8.3f} {np.percentile(ratio,25):7.3f} {np.percentile(ratio,75):7.3f}")
xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
good = ys > 0.04
if good.sum() >= 4:
    w = ns[good].astype(float); z = -np.log(np.clip(ys[good],1e-4,1.0))
    A = np.vstack([xs[good]**2, np.ones(good.sum())]).T
    W = np.diag(w); coef = np.linalg.lstsq(A.T@W@A, A.T@W@z, rcond=None)[0]
    if coef[0] > 0:
        sigma = np.sqrt(1.0/(2*coef[0])); lam0 = np.exp(coef[1])
        print(f"\nfitted  ratio(t) = {lam0:.3f} * exp(-t^2 / (2*{sigma:.2f}^2))")
        print(f"paper   ratio(t) =  1.000 * exp(-t^2 / (2*15.00^2))   (sigma_v = pi/12)")
        print(f"        -> shipped decay is {15.0/sigma:.2f}x faster than the paper's sigma_v")
        print(f"        -> and {12.0/sigma:.2f}x faster than the paper's sigma_h")
print("\n--- roll handling (Level vs Full vs Auto) ---")
rm = np.abs(c["rz_full"]) > 1.0
if rm.sum():
    print(f"n={int(rm.sum())}  median |rz|: full={np.median(np.abs(c['rz_full'][rm])):.3f} auto={np.median(np.abs(c['rz_auto'][rm])):.3f} level={np.median(np.abs(c['rz_level'][rm])):.3f}")
    rr = np.abs(c["rz_auto"][rm])/np.maximum(np.abs(c["rz_full"][rm]),1e-9)
    print(f"median auto/full ROLL ratio = {np.median(rr):.3f}")
    if m.sum(): print(f"median auto/full KEYSTONE ratio = {np.median(k_auto[m]/np.maximum(k_full[m],1e-9)):.3f}")
print("\n--- Vertical mode semantics ---")
vm = k_full > 3.0
if vm.sum():
    print(f"n={int(vm.sum())}")
    print(f"  pitch |rx|: full={np.median(np.abs(c['rx_full'][vm])):7.3f} vert={np.median(np.abs(c['rx_vert'][vm])):7.3f} auto={np.median(np.abs(c['rx_auto'][vm])):7.3f}")
    print(f"  yaw   |ry|: full={np.median(np.abs(c['ry_full'][vm])):7.3f} vert={np.median(np.abs(c['ry_vert'][vm])):7.3f} auto={np.median(np.abs(c['ry_auto'][vm])):7.3f}")
print("\n--- Level mode semantics ---")
print(f"  n={len(c['img'])}  median |rx_level|={np.median(np.abs(c['rx_level'])):.3f} |ry_level|={np.median(np.abs(c['ry_level'])):.3f} |rz_level|={np.median(np.abs(c['rz_level'])):.3f}")
print(f"  median |rx_full|={np.median(np.abs(c['rx_full'])):.3f} |ry_full|={np.median(np.abs(c['ry_full'])):.3f} |rz_full|={np.median(np.abs(c['rz_full'])):.3f}")
np.savez("/tmp/lambda_curve.npz", xs=xs, ys=ys, ns=ns)
