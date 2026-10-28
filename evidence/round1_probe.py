"""Round 1 causal probes: reconstruct only the old selector in the repaired solver.

Run from upright/: python3.12 evidence/round1_probe.py. Candidate axis scores
measure the attempted undamped fit even when the current gate returns identity.
The historical public outputs are preserved separately in round1_before.json.
"""
from pathlib import Path
import sys
import inspect
import json
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'port'))
import upright_port as up
import benchmark_accuracy as bench


def raw_angles(H, seg, w, h):
    lines = up._homogeneous_lines(seg) @ np.linalg.inv(H)
    dx, dy = -lines[:, 1] * w, lines[:, 0] * h
    return np.degrees(np.arctan2(np.abs(dx), np.abs(dy))), np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))


def run_probe():
    im = cv2.imread(str(ROOT / 'lrtest/in/D_roll.tif'), 0)
    h, w = im.shape
    seg = up.detect_segments(im)
    truth = bench.ground_truth(bench.CASES['D_roll'], 1600, w, h, 9, -5, 20)
    records = {}
    lm = up._levenberg_marquardt
    good_start = None

    def run(label, seed, start=None, segments=seg):
        def traced(fun, x0, lo, hi, **kwargs):
            nonlocal good_start
            if start is not None:
                x0 = start.copy()
            theta, cost = lm(fun, x0, lo, hi, **kwargs)
            if label == 'seed0':
                good_start = theta.copy()
            context = inspect.getclosurevars(fun).nonlocals
            K1 = np.array([[np.exp(theta[3])/w, 0, theta[4]], [0, np.exp(theta[3])/h, theta[5]], [0,0,1]])
            H = K1 @ up._rotation(*theta[:3]) @ context['K0i']
            parts = {}
            for family, axis in [('vert', 0), ('horiz', 1)]:
                lines, vp = context[family]
                angles = raw_angles(H, lines, w, h)[axis]
                lengths = np.hypot((lines[:,2]-lines[:,0])*w, (lines[:,3]-lines[:,1])*h)
                parts[family] = dict(vp=vp.tolist(), count=len(lines), median_deg=float(np.median(angles)),
                                     rms_deg=float(np.sqrt(np.average(angles**2, weights=lengths))))
            records[label] = dict(seed=seed, x0=x0.tolist(), theta=theta.tolist(), cost=cost, families=parts,
                                  segments=len(segments),
                                  axis=bench.deviation(H, truth, np.array([0,0,1,1]), w, h),
                                  quality=up._alignment_quality(H, context['vert'][0], context['horiz'][0], w, h),
                                  depth=None if context['vp_h2'] is None else context['vp_h2'].tolist())
            return theta, cost
        with patch.object(up, '_levenberg_marquardt', traced):
            out = up.solve_ex(segments, w, h, 1600, 'full', seed=seed)
        records[label].update(ok=out['ok'], reason=out['reason'],
                              returned_axis=bench.deviation(out['H'],truth,np.array([0,0,1,1]),w,h))

    source = inspect.getsource(up._base_solve)
    current_score = 'return abs(_normalize(K0i @ v["vp"])[0])'
    assert current_score in source
    source = source.replace(current_score,
        'p = v["vp"]\n        n = float(np.hypot(p[0], p[1]))\n        return abs(p[0]) / n if n > _EPS else 0.0')
    namespace = dict(up.__dict__)
    exec(source, namespace)
    namespace['_levenberg_marquardt'] = lambda *a, **kw: up._levenberg_marquardt(*a, **kw)

    # Reconstruct the rejected review trial: no proxy bin gates, two-pass NFA budget.
    log_nfa = up._log_nfa
    with patch.object(up, '_significance_bins', return_value=0), patch.object(
            up, '_log_nfa', side_effect=lambda *a: log_nfa(*a) + np.log(2)):
        trial = up.detect_segments(im)
    with patch.object(up, '_base_solve', namespace['_base_solve']):
        run('seed0', 0)
        run('seed4', 4)
        run('seed4_good_start_same_bad_families', 4, good_start)
        with np.load(bench.SEGMENTS) as frozen:
            run('seed4_old_detector', 4, segments=frozen['D_roll'])
        run('trial_seed9_old_selector', 9, segments=trial)
    run('seed4_camera_centered_selection', 4)
    run('trial_seed9_camera_centered_selection', 9, segments=trial)
    (ROOT / 'evidence/round1_diagnosis.json').write_text(json.dumps(records, indent=2) + '\n')
    for key, row in records.items():
        print(key, 'axis', row['axis'], 'cost', row['cost'], 'family', row['families'])


if __name__ == '__main__':
    run_probe()
