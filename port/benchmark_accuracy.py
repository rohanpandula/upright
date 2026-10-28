"""Run with Python 3.12. Compare the port against frozen pre-change results.

MEASURED fixtures/protocol: evidence/accuracy_gap.txt. The four TIFF poses and
1600 px focal are from make_test_images.py, CASES in validate_vs_adobe.py.
01_input.png is ALSO synthetic: demo_synthetic.py specifies its 1400 px focal.
Score ground-truth world lines, independently of either detector. A separate
legacy-inlier score reproduces the status note's incorrect 1200 px focal.
"""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np

import upright_port as up

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'evidence' / 'accuracy_baseline.json'
SEGMENTS = ROOT / 'evidence' / 'accuracy_baseline_segments.npz'

# MEASURED generating poses, make_test_images.py (radians, not rounded degrees).
CASES = dict(A_clean=(-0.384, 0.105, 0.0), B_strong=(-0.60, 0.21, 0.0),
             C_noisy=(-0.384, 0.105, 0.0), D_roll=(-0.384, 0.105, 0.052))


def deviation(Hn, segments, kind, w, h):
    """Median absolute axis deviation, in isotropic pixel coordinates."""
    N = np.diag([1.0 / w, 1.0 / h, 1.0])
    Hp = np.linalg.inv(N) @ Hn @ N
    lines = up._homogeneous_lines(segments) @ np.linalg.inv(Hp)
    angle = np.degrees(np.arctan2(np.abs(lines[:, 0]), np.abs(lines[:, 1])))
    return [float(np.median(90 - angle[kind == 0])),
            float(np.median(angle[kind == 1]))]


def candidate_deviation(out, segments, kind, w, h):
    """Score the undamped candidate even when the gate returns identity.

    Isotropic destination focal and translation leave pixel angles unchanged.
    """
    if out.get('rotation') is None:
        return None
    K = up._intrinsics_norm(out['source_focal_px'], w, h)
    H = K @ up._rotation(*out['rotation']) @ np.linalg.inv(K)
    return deviation(H, segments, kind, w, h)


def ground_truth(pose, focal, w, h, x, ylo, yhi):
    R = up._rotation(*pose)
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    world = [[(-x, ylo, 25), (-x, yhi, 25)],
             [(x, ylo, 25), (x, yhi, 25)],
             [(-x, ylo, 25), (x, ylo, 25)],
             [(-x, yhi, 25), (x, yhi, 25)]]
    p = np.array(world) @ R @ K.T
    return (p[:, :, :2] / p[:, :, 2:]).reshape(-1, 4)


def run(record_baseline=False):
    rows = {}
    saved = {}
    if not record_baseline:
        with np.load(SEGMENTS) as frozen:
            saved = {key: frozen[key] for key in frozen.files}
    fixtures = [(name, ROOT / 'lrtest' / 'in' / (name + '.tif'), pose,
                 1600.0, 9, -5, 20) for name, pose in CASES.items()]
    fixtures.append(('01_input_truth', ROOT / '01_input.png',
                     (-0.38, 0.10, 0.04), 1400.0, 8, -5, 18))
    for name, path, pose, focal, x, ylo, yhi in fixtures:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape
        truth = ground_truth(pose, focal, w, h, x, ylo, yhi)
        kind = np.array([0, 0, 1, 1])
        detected = up.detect_segments(image)
        if record_baseline:
            saved[name] = detected
        rows[name] = {}
        for pipeline, seg in [('fixed_segments', saved[name]), ('image', detected)]:
            result = up.solve_ex(seg, w, h, focal, 'full')
            if record_baseline:
                saved[name + '_H'] = result['H']
            rows[name][pipeline] = dict(ok=result['ok'], segments=len(seg),
                median_axis_deg=deviation(result['H'], truth, kind, w, h),
                candidate_axis_deg=candidate_deviation(result, truth, kind, w, h))
        if name == '01_input_truth':
            # Same world oracle as the correct-focal row; keep the historical
            # detector-inlier score below unchanged and explicitly separate.
            rows['01_input_wrong_focal_truth'] = {}
            for pipeline, seg in [('fixed_segments', saved[name]), ('image', detected)]:
                result = up.solve_ex(seg, w, h, float(w), 'full')
                rows['01_input_wrong_focal_truth'][pipeline] = dict(
                    ok=result['ok'], segments=len(seg),
                    median_axis_deg=deviation(result['H'], truth, kind, w, h),
                    source_focal_px=result['source_focal_px'])
            if record_baseline:
                vps = up.estimate_vanishing_points(detected, w, h)
                vertical, horizontal = up._classify(vps, detected / [w, h, w, h])
                saved['legacy_v'] = detected[vertical['inliers']]
                saved['legacy_h'] = detected[horizontal[0]['inliers']]
            legacy = np.concatenate([saved['legacy_v'], saved['legacy_h']])
            kind = np.r_[np.zeros(len(saved['legacy_v'])),
                         np.ones(len(saved['legacy_h']))]
            rows['01_input_legacy_wrong_focal'] = {}
            for pipeline, seg in [('fixed_segments', saved[name]), ('image', detected)]:
                result = up.solve_ex(seg, w, h, float(w), 'full')
                if record_baseline:
                    saved['legacy_H'] = result['H']
                rows['01_input_legacy_wrong_focal'][pipeline] = dict(
                    ok=result['ok'], segments=len(seg),
                    median_axis_deg=deviation(result['H'], legacy, kind, w, h))
    if record_baseline:
        np.savez_compressed(SEGMENTS, **saved)
        BASELINE.write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def run_heldout(seeds=(0,)):
    """Separate image-only benchmark. Include returned identity on every failure."""
    import sys
    sys.path.insert(0, str(ROOT))
    import make_heldout_images as heldout
    rows = {}
    for case in heldout.CASES:
        image, truth, kind, oracle = heldout.render(case)
        w, h = case['w'], case['h']
        focal = float(heldout.camera(case)[1][0, 0])
        segments = up.detect_segments(image)
        trials = []
        for seed in seeds:
            out = up.solve_ex(segments, w, h, focal, 'full', seed=seed)
            trials.append(dict(seed=seed, ok=out['ok'], reason=out['reason'],
                median_axis_deg=deviation(out['H'], truth, kind, w, h),
                source_focal_px=out['source_focal_px'], calibration=out['calibration'],
                quality=out['quality']))
        rows[case['name']] = dict(camera=case, segments=len(segments),
            truth_counts=[int(np.sum(kind==i)) for i in range(3)],
            uncorrected_axis_deg=deviation(np.eye(3), truth, kind, w, h),
            oracle_axis_deg=deviation(oracle, truth, kind, w, h), trials=trials)
    scores = np.array([trial['median_axis_deg'] for row in rows.values()
                       for trial in row['trials']])
    if not np.all(np.isfinite(scores)):
        raise ValueError('held-out oracle or returned score is non-finite')
    return dict(protocol='held-out NumPy ray/plane raster; full; true metadata; '
                'independent visible world lines; failed returned identities included',
                scenes=rows, mean_family_deg=float(scores.mean()),
                worst_family_deg=float(scores.max()),
                failures=sum(not t['ok'] for r in rows.values() for t in r['trials']),
                attempts=len(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-baseline', action='store_true',
                        help='Use ONLY with the original port: replaces frozen baseline')
    parser.add_argument('--mode', choices=('fixtures', 'heldout'), default='fixtures')
    parser.add_argument('--seeds', type=int, default=1, help='held-out seeds 0..N-1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'heldout':
        if args.record_baseline or args.seeds < 1:
            parser.error('heldout needs positive --seeds and cannot replace the fixture baseline')
        results = run_heldout(range(args.seeds))
        print('Held-out data ONLY (separate from fixture mean)')
        print(f"{'scene':24} seed  returned V/H deg      gate  support bias bound V/H deg")
        for name, row in results['scenes'].items():
            for t in row['trials']:
                v, h = t['median_axis_deg']
                q = t['quality']
                bounds = q.get('family_bias_bound_deg') if q else None
                print(f"{name:24} {t['seed']:4d}  {v:9.6f}/{h:9.6f}  "
                      f"{'PASS' if t['ok'] else 'FAIL':4}  {bounds}")
        print(f"mean family {results['mean_family_deg']:.6f} deg; "
              f"worst {results['worst_family_deg']:.6f}; "
              f"failures {results['failures']}/{results['attempts']}")
        target = args.output or ROOT / 'evidence' / 'heldout_accuracy.json'
        target.write_text(json.dumps(results, indent=2)+'\n')
        raise SystemExit(0)
    rows = run(args.record_baseline)
    if args.record_baseline:
        print(json.dumps(rows, indent=2))
    else:
        before = json.loads(BASELINE.read_text())
        print(f"{'scene':32} {'before V/H':>15} {'energy only V/H':>17} {'after V/H':>15}")
        for name, row in rows.items():
            def pair(result):
                return ' / '.join(f'{value:.4f}' for value in result['median_axis_deg'])
            prior = pair(before[name]['image']) if name in before else 'n/a (new oracle)'
            print(f"{name:32} {prior:>15} "
                  f"{pair(row['fixed_segments']):>17} {pair(row['image']):>15}")
        (args.output or ROOT / 'evidence' / 'accuracy_after.json').write_text(
            json.dumps(rows, indent=2) + '\n')
