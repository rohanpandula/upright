"""MIT clean-room review experiment. Run: python3.12 port/benchmark_sensitivity.py.

Change only the family loss/scale/length weighting. Freeze each detector's output
before solving. Reconstruct the pre-review detector gates and verify its default
results against port_review_pre_fix_accuracy.json, so the 92% claim is tested too.
"""
from pathlib import Path
import argparse
import importlib.util
from unittest.mock import patch
import json

import cv2
import numpy as np

import benchmark_accuracy as benchmark
import upright_port as up

ROOT = Path(__file__).resolve().parents[1]


def family_terms(loss, factor=1.0, length_mode='mean'):
    def terms(H, seg_n, axis, weight, w, h, mean_length):
        lines = up._homogeneous_lines(seg_n) @ np.linalg.inv(H)
        d = up._normalize_rows(np.column_stack([-lines[:, 1] * w, lines[:, 0] * h]))
        r = d @ axis
        s = factor * 4.0 / (max(w, h) - 1)
        if loss == 'cauchy':
            residual = np.sign(r) * s * np.sqrt(np.log1p((r / s) ** 2))
        elif loss == 'huber':
            rho = np.where(np.abs(r) <= s, r * r, 2 * s * np.abs(r) - s * s)
            residual = np.sign(r) * np.sqrt(rho)
        elif loss == 'least_squares':
            residual = r
        else:
            raise ValueError(loss)
        lengths = np.hypot((seg_n[:, 2] - seg_n[:, 0]) * w,
                           (seg_n[:, 3] - seg_n[:, 1]) * h)
        weighted = weight * lengths / max(mean_length, up._EPS)
        if length_mode == 'uniform':
            weighted = np.full(len(seg_n), weight)
        elif length_mode == 'half_mean':
            weighted *= 2
        elif length_mode == 'double_mean':
            weighted *= 0.5
        elif length_mode != 'mean':
            raise ValueError(length_mode)
        return np.sqrt(weighted) * residual
    return terms


def pre_review_detection(image):
    """The unchanged original smoothed pass, with the new fallback disabled."""
    detect = up._detect_gradient_segments
    with patch.object(up, '_detect_gradient_segments', side_effect=lambda image, sigma:
                      detect(image, sigma) if sigma else np.empty((0, 4))):
        return up.detect_segments(image)


def run(output_path=None, record_failures=False):
    reference = family_terms('cauchy')
    seg = np.array([[0.1, 0.2, 0.3, 0.7], [0.1, 0.2, 0.100001, 0.6]])
    for w, h in ((1400, 1000), (1000, 1400)):
        for axis in (np.array([1., 0.]), np.array([0., 1.])):
            np.testing.assert_allclose(reference(np.eye(3), seg, axis, 2, w, h, 30),
                                       up._family_terms(np.eye(3), seg, axis, 2, w, h, 30),
                                       rtol=1e-14, atol=1e-14)
    # Replay the frozen implementation: even homogeneous VP renormalization
    # changes finite-difference roundoff in its unidentifiable centre parameters.
    spec = importlib.util.spec_from_file_location('round3_original',
        ROOT / 'evidence/round3_original_port.py')
    historical_port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical_port)
    old_after = json.loads((ROOT / 'evidence/port_review_pre_fix_accuracy.json').read_text())
    baseline = json.loads(benchmark.BASELINE.read_text())
    fixtures = []
    with np.load(benchmark.SEGMENTS) as frozen:
        for name, pose in [*benchmark.CASES.items(),
                           ('01_input_truth', (-0.38, 0.10, 0.04))]:
            special = name == '01_input_truth'
            path = ROOT / '01_input.png' if special else ROOT / 'lrtest/in' / (name + '.tif')
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape
            focal = 1400.0 if special else 1600.0
            truth = benchmark.ground_truth(pose, focal, w, h, 8 if special else 9,
                                           -5, 18 if special else 20)
            detectors = dict(original=frozen[name], pre_review=pre_review_detection(image),
                             review=up.detect_segments(image))
            fixtures.append((name, w, h, focal, truth, np.array([0, 0, 1, 1]), detectors))
            if special:
                legacy = np.concatenate([frozen['legacy_v'], frozen['legacy_h']])
                kinds = np.r_[np.zeros(len(frozen['legacy_v'])), np.ones(len(frozen['legacy_h']))]
                fixtures.append(('01_input_legacy_wrong_focal', w, h, float(w),
                                 legacy, kinds, detectors))
    variants = [(loss, scale, 'mean') for loss in ('cauchy', 'huber')
                for scale in (0.5, 1.0, 2.0)]
    variants += [('least_squares', 1.0, 'mean')]
    variants += [('cauchy', 1.0, length) for length in ('uniform', 'half_mean', 'double_mean')]
    rows = {}
    alignment_quality = up._alignment_quality
    optimizer = up._levenberg_marquardt
    candidate = {}

    def historical_optimizer(fun, x0, lo, hi):
        lo, hi = lo.copy(), hi.copy()
        lo[4:], hi[4:] = 0.2, 0.8
        return optimizer(fun, x0, lo, hi)

    def capture_quality(H, *args):
        candidate['H'] = H.copy()
        return alignment_quality(H, *args)

    for loss, scale, length in variants:
        key = f'{loss}_{scale:g}_{length}'
        rows[key] = {}
        with patch.object(up, '_family_terms', family_terms(loss, scale, length)), \
                patch.object(up, '_alignment_quality', capture_quality):
            for name, w, h, focal, truth, kind, detectors in fixtures:
                rows[key][name] = {}
                for detector, detected in detectors.items():
                    candidate.clear()
                    result = up.solve_ex(detected, w, h, focal, 'full', seed=0)
                    if not result['ok'] and not record_failures:
                        raise AssertionError((key, name, detector, result['reason']))
                    score = benchmark.deviation(result['H'], truth, kind, w, h)
                    rows[key][name][detector] = dict(median_axis_deg=score,
                        segments=len(detected), ok=result['ok'], reason=result['reason'],
                        quality=result['quality'], candidate_axis_deg=(
                            benchmark.deviation(candidate['H'], truth, kind, w, h)
                            if 'H' in candidate else None))
                    if key == 'cauchy_1_mean' and detector == 'pre_review':
                        # Restore historical source, VP, gate and centre policies;
                        # the current candidate and rejection are recorded above.
                        with patch.object(historical_port, '_source_calibration',
                                side_effect=lambda *a: (a[-1], None)), patch.object(
                                historical_port, '_levenberg_marquardt', side_effect=historical_optimizer):
                            historical = historical_port.solve_ex(detected,w,h,focal,'full',seed=0)
                        historical_score = benchmark.deviation(
                            historical['H'],truth,kind,w,h)
                        np.testing.assert_allclose(historical_score, old_after[name]['image']['median_axis_deg'],
                                                   rtol=0, atol=1e-7)
        print(key, 'candidate mean', {detector: round(float(np.mean([
            rows[key][name][detector]['candidate_axis_deg'] for name in benchmark.CASES])), 6)
            for detector in ('pre_review', 'review', 'original')},
            'rejected', sum(not r['ok'] for scene in rows[key].values() for r in scene.values()),
            flush=True)
    output = dict(protocol='Full, seed=0, true focal except explicit legacy row; '
                           'frozen detections, independent world lines; degrees',
                  baseline=baseline, variants=rows)
    path = Path(output_path) if output_path is not None else ROOT / 'evidence/port_review_sensitivity.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--record-failures', action='store_true',
                        help='Record rejected candidates and returned identity scores without aborting')
    args = parser.parse_args()
    run(args.output, args.record_failures)
