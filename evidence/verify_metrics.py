"""Independent verification of port clone-distance claims. MIT.

Reimplements the 625-point RMS metric from the lua blobs and scene TIFFs.
Does not import any benchmark script. Run: python3.12 evidence/verify_metrics.py
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'port'))
sys.path.insert(0, str(ROOT / 'evidence'))
import upright_port as up  # noqa: E402

W_PX, H_PX = 1400, 1000
F = 1600.0
SLOTS = {'auto': 1, 'full': 2, 'level': 3, 'vertical': 4}
CASES = {  # exact generating poses, radians
    'A_clean': dict(img=6727297, rx=-0.384, ry=0.105, rz=0.0),
    'B_strong': dict(img=6727497, rx=-0.60, ry=0.21, rz=0.0),
    'C_noisy': dict(img=6727693, rx=-0.384, ry=0.105, rz=0.0),
    'D_roll': dict(img=6727887, rx=-0.384, ry=0.105, rz=0.052),
}


def parse_lua(path):
    out = {}
    for m in re.finditer(r'UprightTransform_(\d)\s*=\s*"([^"]+)"', path.read_text()):
        v = [float(x) for x in m.group(2).split(',')]
        if len(v) == 9:
            out[int(m.group(1))] = np.array(v).reshape(3, 3)
    return out


def grid(H):
    """625 fixed source samples on the 25x25 grid, in pixel units."""
    H = np.asarray(H, float)
    H = H / H[2, 2]
    yy, xx = np.mgrid[0:1:25j, 0:1:25j]
    p = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    q = (H @ p.T).T
    if np.any(q[:, 2] <= 0):
        raise ValueError('horizon crossing')
    return q[:, :2] / q[:, 2:] * [W_PX, H_PX]


def score(aH, bH):
    a, b = grid(aH), grid(bH)
    raw = float(np.sqrt(np.mean(np.sum((a - b) ** 2, 1))))
    ac, bc = a - a.mean(0), b - b.mean(0)
    s = float(np.sum(ac * bc) / np.sum(bc * bc))
    shape = float(np.sqrt(np.mean(np.sum((ac - s * bc) ** 2, 1))))
    unit = float(np.hypot(W_PX, H_PX) / 100)
    return dict(raw=raw / unit, shape=shape / unit, applied_or_fit_scale=s)


def centre_lb(adobe, framed):
    """Best single scale about (0.5,0.5), translation framing held fixed."""
    a, b = grid(adobe), grid(framed)
    c = np.array([W_PX / 2, H_PX / 2])
    d = b - c
    # minimise sum |(a-c) - s*d|^2
    s = float(np.sum((a - c) * d) / np.sum(d * d))
    resid = np.sqrt(np.mean(np.sum(((a - c) - s * d) ** 2, 1)))
    return resid / np.hypot(W_PX, H_PX) * 100


def rotation(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def main():
    seeds = range(5)
    rows = []
    for name, case in CASES.items():
        adobe = parse_lua(ROOT / 'adobe_truth' / f"img_{case['img']}.lua")
        image = cv2.imread(str(ROOT / 'lrtest' / 'in' / (name + '.tif')), 0)
        h, w = image.shape
        assert (w, h) == (W_PX, H_PX)
        seg = up.detect_segments(image)
        K = up._intrinsics_norm(F, w, h)
        Ki = np.linalg.inv(K)
        naive = K @ rotation(case['rx'], case['ry'], case['rz']) @ Ki
        naive /= naive[2, 2]
        naive_pol = up.scale_output(up.frame_output(naive))
        for mode, slot in SLOTS.items():
            for seed in seeds:
                aH = adobe[slot]
                out = up.solve_ex(seg, w, h, F, mode, seed=seed,
                                  framing=True, scaling=True)
                unframed = up.solve_ex(seg, w, h, F, mode, seed=seed)
                framed = up.frame_output(unframed['H'])
                # mode-aware oracle: true rotation, port's own damping vector
                damp = out['damping']
                if out['rotation'] is not None and damp is not None:
                    r = case
                    Ro = rotation(r['rx'] * damp[0], r['ry'] * damp[1],
                                  r['rz'] * damp[2])
                    # port's destination camera, perfect rotation: isolates
                    # solver geometry from the output policy
                    K0 = up._intrinsics_norm(out['source_focal_px'], w, h)
                    Rd = rotation(*out['rotation'])
                    Rd = rotation(out['rotation'][0] * damp[0],
                                  out['rotation'][1] * damp[1],
                                  out['rotation'][2] * damp[2])
                    K1 = out['H'] @ K0 @ np.linalg.inv(Rd)
                    oracle_mode = up.scale_output(up.frame_output(K1 @ Ro @ Ki))
                else:
                    oracle_mode = naive_pol
                rows.append(dict(
                    scene=name, mode=mode, seed=seed, ok=out['ok'],
                    port=score(aH, out['H'])['raw'],
                    framed=score(aH, framed)['raw'],
                    unframed=score(aH, unframed['H'])['raw'],
                    shape=score(aH, unframed['H'])['shape'],
                    centre_lb=centre_lb(aH, framed),
                    naive=score(aH, naive)['raw'],
                    naive_shape=score(aH, naive)['shape'],
                    naive_pol=score(aH, naive_pol)['raw'],
                    oracle_mode=score(aH, oracle_mode)['raw'],
                    applied_scale=out['H'][0, 0] / unframed['H'][0, 0],
                ))
                print(name, mode, seed, f"{rows[-1]['port']:.6f}", flush=True)

    def mean(key, sub=None):
        s = sub if sub is not None else rows
        return float(np.mean([r[key] for r in s]))

    summary = {}
    summary['n'] = len(rows)
    for key in ('port', 'framed', 'unframed', 'shape', 'centre_lb',
                'naive', 'naive_shape', 'naive_pol', 'oracle_mode'):
        summary[key] = mean(key)
    summary['by_mode'] = {}
    for mode in SLOTS:
        sub = [r for r in rows if r['mode'] == mode]
        summary['by_mode'][mode] = {k: mean(k, sub) for k in
                                    ('port', 'shape', 'naive_pol', 'naive_shape',
                                     'oracle_mode')}
    summary['rows'] = rows
    (ROOT / 'evidence' / 'verify_metrics.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
