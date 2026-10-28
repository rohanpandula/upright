"""Re-run the review's structure and noise controls without regenerating rasters."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'port'), str(ROOT)]
import upright_port as up
import benchmark_accuracy as bench
import make_test_images as scenes


def clip(a, b, w, h):
    delta = b - a
    lo, hi = 0.0, 1.0
    for i, extent in enumerate((w - 1, h - 1)):
        if abs(delta[i]) < 1e-12:
            if not 0 <= a[i] <= extent:
                return None
        else:
            t = sorted((-a[i] / delta[i], (extent - a[i]) / delta[i]))
            lo, hi = max(lo, t[0]), min(hi, t[1])
            if lo > hi:
                return None
    return np.r_[a + lo * delta, a + hi * delta]


def run():
    structure = {}
    before = json.loads((ROOT / 'evidence/port_review_structure.json').read_text())
    for name in before:
        image = cv2.imread(str(ROOT / before[name]['path']), 0)
        h, w = image.shape
        K = np.array([[1600, 0, w/2], [0, 1600, h/2], [0, 0, 1]])
        R = up._rotation(scenes.PITCH15, scenes.YAW6, 0)
        truth, kind = [], []
        for a, b, _, family in scenes.SCENES[name]()[0]:
            if family not in ('v', 'h'):
                continue
            points = np.array([a, b]) @ R @ K.T
            if np.min(points[:, 2]) <= 0.2:
                continue
            points = points[:, :2] / points[:, 2:]
            segment = clip(*points, w, h)
            if segment is not None and np.linalg.norm(segment[2:] - segment[:2]) >= 20:
                truth.append(segment)
                kind.append(0 if family == 'v' else 1)
        counts = [kind.count(0), kind.count(1)]
        assert counts == before[name]['truth_lines'], (name, counts)
        result = up.solve_ex(image=image, focal_px=1600, mode='full')
        structure[name] = dict(ok=result['ok'], reason=result['reason'], quality=result['quality'],
            segments=len(result['segments']), truth_lines=counts,
            before_axis=before[name]['median_axis_deg'],
            after_axis=bench.deviation(result['H'],np.array(truth),np.array(kind),w,h))
    noise = {}
    for size in (32, 64, 128, 256):
        counts = [len(up.detect_segments(np.random.default_rng(seed).integers(
            0, 256, (size, size), np.uint8))) for seed in range(30)]
        noise[size] = dict(counts=counts, mean=float(np.mean(counts)), max=max(counts),
                           nonempty=int(np.count_nonzero(counts)))
    result = dict(structure=structure, noise=noise)
    (ROOT / 'evidence/round1_extra.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
