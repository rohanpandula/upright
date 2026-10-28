"""
Structural study of Adobe's real Upright output, harvested from a live Lightroom
catalog. No image loading required -- this reads what the shipped binary wrote.

Tests the specific structural claims of the reverse engineering:

  C1  Level (slot 3) is a *pure rotation*: its homography's third row is exactly
      (0, 0, 1), i.e. H = K R K^-1 with no z-tilt.  This is the signature of
      "roll only".
  C2  Vertical (slot 4) is *single-axis*: r20 != 0 and r21 == 0 (pitch only).
  C3  Full (slot 2) and Auto (slot 1) are two-axis: both r20 and r21 non-zero.
  C4  Auto is deliberately *weaker* than Full (adaptive weights, not a max-out).
  C5  Adobe abstains (identity) on a meaningful share of photos.
  C6  The focal length is estimated per photo, not left at the 35mm default.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IDENT = np.eye(3)
SLOT_NAMES = {0: "off", 1: "auto", 2: "full", 3: "level", 4: "vertical", 5: "guided"}


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            recs.append(json.loads(line))
    return recs


def hm(rec, slot):
    v = rec["transform"].get(str(slot))
    return None if v is None else np.array(v, dtype=float).reshape(3, 3)


def is_ident(H, tol=1e-6):
    return np.max(np.abs(H - IDENT)) < tol


def perspective_mag(H):
    """Scale-invariant size of the keystone part, |(h20,h21)| / h22."""
    if abs(H[2, 2]) < 1e-12:
        return float("nan")
    return float(np.hypot(H[2, 0], H[2, 1]) / abs(H[2, 2]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lr_all.jsonl"
    recs = load(path)
    print(f"loaded {len(recs)} analysed photos\n")

    # ---- availability / abstention -------------------------------------
    print("== slot availability and identity (abstention) rates ==")
    for slot in range(6):
        have = sum(1 for r in recs if str(slot) in r["transform"])
        ident = sum(1 for r in recs if str(slot) in r["transform"] and is_ident(hm(r, slot)))
        pct = 100.0 * ident / have if have else 0
        print(f"  slot {slot} ({SLOT_NAMES[slot]:<8}) present {have:6d}   identity {ident:6d} ({pct:5.1f}%)")

    # ---- C1 level is a pure rotation ------------------------------------
    print("\n== C1: Level (slot 3) third row == (0,0,1) ? ==")
    tot = ok = 0
    maxdev = 0.0
    for r in recs:
        H = hm(r, 3)
        if H is None:
            continue
        tot += 1
        dev = float(np.max(np.abs(H[2, :] - np.array([0.0, 0.0, 1.0]))))
        maxdev = max(maxdev, dev)
        if dev < 1e-6:
            ok += 1
    print(f"  exactly affine: {ok}/{tot} = {100.0*ok/tot:.2f}%   max deviation: {maxdev:.3e}")

    # ---- C2/C3 axis structure -------------------------------------------
    print("\n== C2/C3: which perspective axis is active per mode? ==")
    print(f"  {'slot':<10} {'n':>6} {'|h20|>1e-5':>12} {'|h21|>1e-5':>12} {'both':>8} {'x only':>9} {'y only':>9}")
    for slot in (1, 2, 3, 4):
        n = x = y = both = xo = yo = 0
        for r in recs:
            H = hm(r, slot)
            if H is None or is_ident(H):
                continue
            n += 1
            hx, hy = abs(H[2, 0]) > 1e-5, abs(H[2, 1]) > 1e-5
            x += hx; y += hy
            if hx and hy: both += 1
            elif hx: xo += 1
            elif hy: yo += 1
        print(f"  {SLOT_NAMES[slot]:<10} {n:6d} {x:12d} {y:12d} {both:8d} {xo:9d} {yo:9d}")

    # ---- C4 Auto is weaker than Full ------------------------------------
    print("\n== C4: Auto (slot 1) vs Full (slot 2) keystone magnitude ==")
    ratios = []
    for r in recs:
        a, f = hm(r, 1), hm(r, 2)
        if a is None or f is None:
            continue
        pa, pf = perspective_mag(a), perspective_mag(f)
        if not np.isfinite(pa) or not np.isfinite(pf) or pf < 1e-6:
            continue
        ratios.append(pa / pf)
    ratios = np.array(ratios)
    print(f"  n={len(ratios)}  mean={ratios.mean():.4f}  median={np.median(ratios):.4f}")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:02d} = {np.percentile(ratios, q):.4f}")
    print(f"  Auto strictly weaker than Full: {100.0*np.mean(ratios < 1.0):.1f}% of photos")
    print(f"  Auto == Full (identical):       {100.0*np.mean(np.abs(ratios-1) < 1e-6):.1f}%")

    # ---- C5 abstention ---------------------------------------------------
    print("\n== C5: Full-mode abstention ==")
    print(f"  Full slot identity: {100.0*sum(1 for r in recs if is_ident(hm(r,2)))/len(recs):.1f}%")
    print(f"  Auto slot identity: {100.0*sum(1 for r in recs if is_ident(hm(r,1)))/len(recs):.1f}%")

    # ---- C6 focal estimation ---------------------------------------------
    print("\n== C6: UprightFocalLength35mm (35.0 == untouched default) ==")
    foc = np.array([r["meta"].get("UprightFocalLength35mm", 35.0) for r in recs], dtype=float)
    print(f"  n={len(foc)}  ==35 exactly: {100.0*np.mean(np.abs(foc-35)<1e-9):.1f}%")
    fin = foc[np.abs(foc - 35) >= 1e-9]
    if fin.size:
        print(f"  among the {fin.size} estimated: min={fin.min():.1f} "
              f"median={np.median(fin):.1f} max={fin.max():.1f}")
    cen = np.array([[r["meta"].get("UprightCenterNormX", 0.5),
                     r["meta"].get("UprightCenterNormY", 0.5)] for r in recs])
    off = np.linalg.norm(cen - 0.5, axis=1)
    print(f"  centre offset from (0.5,0.5): median={np.median(off):.4f} p95={np.percentile(off,95):.4f}")


if __name__ == "__main__":
    main()
