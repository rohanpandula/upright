"""Diagnostic: apply the corpus-measured Auto curve through the real solver.

Not shipped. Uses cached Adobe truth for scoring only.
"""
import json, math, sys
sys.path.insert(0, 'port'); sys.path.insert(0, '.'); sys.path.insert(0, 'evidence')
import numpy as np, cv2
import upright_port as up
from benchmark_adobe import agreement
from validate_vs_adobe import CASES, SLOTS, parse_adobe

KX = [1, 3, 5, 7, 9, 11, 13.5, 16.5, 20, 24.5, 31, 40.5]
KY = [0.411, 0.623, 0.644, 0.781, 0.771, 0.740, 0.699, 0.567, 0.581, 0.337, 0.131, 0.058]


def corpus(t_deg):
    return float(np.interp(t_deg, KX, KY))


orig = up.correction_fraction_vertical


def make(offset):
    def f(ks):
        return up._saturate(0.0) if False else corpus(math.degrees(ks) * offset)
    return f


rows = []
for name, case in CASES.items():
    adobe = parse_adobe(f'adobe_truth/img_{case["text"]}.lua')
    img = cv2.imread(f'lrtest/in/{name}.tif', 0)
    h, w = img.shape
    seg = up.detect_segments(img)
    for variant, off in (('port', 0.0), ('c1.0', 1.0), ('c1.4', 1.4), ('c1.6', 1.6), ('c1.8', 1.8)):
        up.correction_fraction_vertical = orig if variant == 'port' else make(off)
        for seed in range(5):
            out = up.solve_ex(seg, w, h, 1600., 'auto', seed=seed)
            ag = agreement(adobe[SLOTS['auto']], out['H'], w, h)
            rows.append(dict(scene=name, variant=variant, seed=seed, ok=out['ok'],
                             raw=ag['raw_rms_pct'], shape=ag['shape_rms_pct'],
                             damp=out['damping'][0] if out['damping'] else None))
    print('done', name, flush=True)
up.correction_fraction_vertical = orig

summary = {}
for name in CASES:
    summary[name] = {}
    for variant in ('port', 'c1.0', 'c1.4', 'c1.6', 'c1.8'):
        rs = [r for r in rows if r['scene'] == name and r['variant'] == variant]
        summary[name][variant] = dict(raw=float(np.mean([r['raw'] for r in rs])),
                                      shape=float(np.mean([r['shape'] for r in rs])),
                                      damp=float(np.mean([r['damp'] for r in rs if r['damp'] is not None])),
                                      rejected=sum(not r['ok'] for r in rs))
for name in summary:
    print(f"{name:9s} " + '  '.join(
        f"{v}: raw={summary[name][v]['raw']:.3f} shape={summary[name][v]['shape']:.3f} "
        f"damp={summary[name][v]['damp']:.3f}" for v in ('port', 'c1.0', 'c1.4', 'c1.6', 'c1.8')))
agg = {v: (float(np.mean([summary[n][v]['raw'] for n in summary])),
           float(np.mean([summary[n][v]['shape'] for n in summary]))) for v in
       ('port', 'c1.0', 'c1.4', 'c1.6', 'c1.8')}
print('\naggregate auto raw/shape:', {k: (round(a, 4), round(b, 4)) for k, (a, b) in agg.items()})
json.dump(dict(summary=summary, rows=rows), open('/tmp/auto_corpus_probe.json', 'w'), indent=1)
