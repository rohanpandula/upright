"""Clean-process seed sweep and bad-optimizer replay. MIT."""
import math
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'port'))
sys.path.insert(0, str(ROOT / 'evidence'))
sys.path.insert(0, str(ROOT))
import upright_port as up  # noqa: E402
import benchmark_accuracy as bench  # noqa: E402

fails = []
print('scene seeds maxV maxH over0.1 failures')
for name, pose in bench.CASES.items():
    image = cv2.imread(str(ROOT / 'lrtest' / 'in' / (name + '.tif')), 0)
    h, w = image.shape
    seg = up.detect_segments(image)
    truth = bench.ground_truth(pose, 1600, w, h, 9, -5, 20)
    vs, hs, bad, rej = [], [], 0, 0
    for seed in range(50):
        out = up.solve_ex(seg, w, h, 1600, 'full', seed=seed)
        v, hh = bench.deviation(out['H'], truth, np.array([0, 0, 1, 1]), w, h)
        vs.append(v); hs.append(hh)
        if not out['ok']:
            rej += 1
        if max(v, hh) >= 0.1:
            bad += 1
            fails.append((name, seed, v, hh, out['reason']))
    print(name, 50, f'{max(vs):.6f}', f'{max(hs):.6f}', bad, rej)
    print(f'  seed4 {vs[4]:.6f}/{hs[4]:.6f} ok={up.solve_ex(seg,w,h,1600,"full",seed=4)["ok"]}')

# Replay the recorded catastrophic optimizer output through the public API.
image = cv2.imread(str(ROOT / 'lrtest' / 'in' / 'D_roll.tif'), 0)
h, w = image.shape
seg = up.detect_segments(image)
theta = np.array([-0.17811796261583085, 1.2, 0.13403823499357556,
                  math.log(1600), 0.8, 0.8])
optimizer = up._levenberg_marquardt
def replay(fun, x0, lo, hi, **kwargs):
    if len(x0) == 4:
        return optimizer(fun, x0, lo, hi, **kwargs)
    return theta, 0.0012678816696807755
print('replay:')
for mode in ('full', 'auto', 'level', 'vertical'):
    with patch.object(up, '_levenberg_marquardt', side_effect=replay):
        out = up.solve_ex(seg, w, h, 1600, mode, seed=4)
    print(' ', mode, out['ok'], out['reason'], np.array_equal(out['H'], np.eye(3)))
print('over0.1:', len(fails))
for f in fails:
    print(' ', f)
