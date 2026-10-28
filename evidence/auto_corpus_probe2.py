"""Diagnostic: corpus curve in camera-pitch units (existing rx argument). Not shipped."""
import json, math, sys
sys.path.insert(0, 'port'); sys.path.insert(0, '.'); sys.path.insert(0, 'evidence')
import numpy as np, cv2
import upright_port as up
from benchmark_adobe import agreement
from validate_vs_adobe import CASES, SLOTS, parse_adobe

KX = [1, 3, 5, 7, 9, 11, 13.5, 16.5, 20, 24.5, 31, 40.5]
KY = [0.532, 0.631, 0.741, 0.815, 0.789, 0.790, 0.692, 0.589, 0.561, 0.468, 0.170, 0.0]


def curve(t, gain):
    return float(np.clip(np.interp(t, KX, KY) * gain, 0.0, 1.0))


orig = up.correction_fraction_vertical
orig_mode = up.mode_damping


def patched(gain):
    def mode(m, rx, ry, rz, *, vertical_keystone_rad=None):
        s = orig_mode(m, rx, ry, rz, vertical_keystone_rad=vertical_keystone_rad)
        if m != 'auto':
            return s
        return (curve(abs(math.degrees(rx)), gain), s[1], s[2])
    return mode


rows = []
for name, case in CASES.items():
    adobe = parse_adobe(f'adobe_truth/img_{case["text"]}.lua')
    img = cv2.imread(f'lrtest/in/{name}.tif', 0)
    h, w = img.shape
    seg = up.detect_segments(img)
    for variant, gain in (('port', 0.0), ('p1.00', 1.0), ('p1.15', 1.15), ('p1.30', 1.30)):
        up.mode_damping = orig_mode if variant == 'port' else patched(gain)
        for seed in range(5):
            out = up.solve_ex(seg, w, h, 1600., 'auto', seed=seed)
            ag = agreement(adobe[SLOTS['auto']], out['H'], w, h)
            rows.append(dict(scene=name, variant=variant, seed=seed, ok=out['ok'],
                             raw=ag['raw_rms_pct'], shape=ag['shape_rms_pct'],
                             damp=out['damping'][0] if out['damping'] else None))
    print('done', name, flush=True)
up.mode_damping = orig_mode

summary = {}
for name in CASES:
    summary[name] = {}
    for v in ('port', 'p1.00', 'p1.15', 'p1.30'):
        rs = [r for r in rows if r['scene'] == name and r['variant'] == v]
        summary[name][v] = dict(raw=float(np.mean([r['raw'] for r in rs])),
                                shape=float(np.mean([r['shape'] for r in rs])),
                                damp=float(np.mean([r['damp'] for r in rs if r['damp'] is not None])),
                                rejected=sum(not r['ok'] for r in rs))
for name in summary:
    print(f"{name:9s} " + '  '.join(
        f"{v}: raw={summary[name][v]['raw']:.3f} shape={summary[name][v]['shape']:.3f} "
        f"damp={summary[name][v]['damp']:.3f}" for v in ('port', 'p1.00', 'p1.15', 'p1.30')))
agg = {v: (float(np.mean([summary[n][v]['raw'] for n in summary])),
           float(np.mean([summary[n][v]['shape'] for n in summary]))) for v in
       ('port', 'p1.00', 'p1.15', 'p1.30')}
print('\naggregate auto raw/shape:', {k: (round(a, 4), round(b, 4)) for k, (a, b) in agg.items()})
json.dump(dict(summary=summary, rows=rows), open('/tmp/auto_corpus_probe2.json', 'w'), indent=1)
