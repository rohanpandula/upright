"""Round 4: compare returned mappings to cached Adobe homographies. MIT.

Run from research/upright: python3.12 port/benchmark_adobe.py --seeds 5
No Adobe transform or world truth enters either solver. Truth is scoring only.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import re
import sys

import cv2
import numpy as np

import upright_port as up
from benchmark_accuracy import deviation, ground_truth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate_vs_adobe import CASES, SLOTS, parse_adobe
from make_test_images import world_segments


def mapped_points(H, w, h):
    """625 fixed source samples, including all four corners, in pixel units."""
    H = np.asarray(H, dtype=float)
    if H[2, 2] == 0:
        raise ValueError('mapping places source origin on the projective horizon')
    H = H / H[2, 2]
    yy, xx = np.mgrid[0:1:25j, 0:1:25j]
    p = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    q = np.einsum('ij,nj->ni', H, p)
    if np.min(q[:, 2]) <= 0:
        raise ValueError('mapping crosses the projective horizon on the source frame')
    q = q[:, :2] / q[:, 2:] * [w, h]
    if not np.all(np.isfinite(q)):
        raise ValueError('non-finite sample')
    return q


def agreement(adobe, candidate, w, h):
    """Raw and best translation/uniform-scale RMS, percent of source diagonal.

    Shape alignment does not remove rotation, shear, anisotropy or perspective.
    Raw agreement retains every difference, including Adobe's output framing.
    """
    a, b = mapped_points(adobe, w, h), mapped_points(candidate, w, h)
    raw = np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))
    ac, bc = a - a.mean(0), b - b.mean(0)
    scale = float(np.sum(ac * bc) / np.sum(bc * bc))
    if scale <= 0:
        raise ValueError('shape alignment would reverse the output')
    shape = np.sqrt(np.mean(np.sum((ac - scale * bc)**2, axis=1)))
    unit = np.hypot(w, h) / 100
    return dict(raw_rms_pct=float(raw / unit), shape_rms_pct=float(shape / unit),
                raw_rms_px=float(raw), shape_rms_px=float(shape),
                fitted_scale=scale, fitted_translation_px=(a.mean(0) - scale*b.mean(0)).tolist())


def visible_verticals(pose, focal, w, h):
    """Independent scoring lines, using the dense-sweep visibility protocol."""
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1.]])
    result = []
    for a, b in world_segments():
        if abs(a[0] - b[0]) > 1e-9 or abs(a[2] - b[2]) > 1e-9:
            continue
        p = np.einsum('ij,nj->ni', K @ up._rotation(*pose).T, np.array([a, b]))
        if np.min(p[:, 2]) <= .1:
            continue
        p = p[:, :2] / p[:, 2:]
        t = np.linspace(0, 1, 41)[:, None]
        samples = (1-t)*p[0] + t*p[1]
        if np.sum(np.all((samples >= [0, 0]) & (samples < [w, h]), axis=1)) >= 8:
            result.append(p.ravel())
    return np.asarray(result) / [w, h, w, h]


def convergence(H, segments, w, h):
    lines = up._homogeneous_lines(segments) @ np.linalg.inv(H)
    tilt = np.arctan2(-lines[:, 1]*w, lines[:, 0]*h)
    tilt = (tilt + np.pi/2) % np.pi - np.pi/2
    return float(np.degrees(np.diff(np.percentile(tilt, [5, 95]))[0]))


def horizon_error(H, pose, w, h):
    """Independent world X/Z horizon tilt after mapping, in pixel angles."""
    K = up._intrinsics_norm(1600, w, h)
    line = np.linalg.inv(K).T @ up._rotation(*pose).T @ np.array([0., 1., 0.])
    line = np.linalg.inv(H).T @ line
    return float(np.degrees(np.arctan2(abs(line[0])*h, abs(line[1])*w)))


def run(seeds):
    spec = importlib.util.spec_from_file_location('round4_original',
                ROOT / 'evidence' / 'round4_original_port.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    rows, scenes = [], {}
    for name, case in CASES.items():
        path = ROOT / 'adobe_truth' / f"img_{case['text']}.lua"
        text = path.read_text()
        adobe = parse_adobe(path)
        assert set(SLOTS.values()) <= set(adobe), path
        metadata = {key: float(re.search(key + r'\s*=\s*([-\d.eE]+)', text)[1])
                    for key in ('UprightFocalLength35mm', 'UprightCenterNormX', 'UprightCenterNormY')}
        image = cv2.imread(str(ROOT / 'lrtest' / 'in' / (name + '.tif')), 0)
        h, w = image.shape
        segments = up.detect_segments(image)
        pose = [case[key] for key in ('rx', 'ry', 'rz')]
        truth = ground_truth(pose, 1600, w, h, 9, -5, 20)
        verticals = visible_verticals(pose, 1600, w, h)
        initial_conv = convergence(np.eye(3), verticals, w, h)
        K = up._intrinsics_norm(1600, w, h)
        naive = K @ up._rotation(*pose) @ np.linalg.inv(K)
        assert horizon_error(naive, pose, w, h) < 1e-8
        scenes[name] = dict(width=w, height=h, segments=len(segments), adobe_metadata=metadata,
                            initial_visible_convergence_deg=initial_conv)
        for prior, focal in (('scene_focal', 1600.), ('adobe_stored_focal',
                 up.focal_px_from_35mm(metadata['UprightFocalLength35mm'], w, h))):
            for seed in seeds:
                for version, solver in (('before', original), ('after', up)):
                    for mode, slot in SLOTS.items():
                        out = solver.solve_ex(segments, w, h, focal, mode, seed=seed)
                        Ha = adobe[slot]
                        # The returned identity on rejection is scored, never dropped.
                        row = dict(scene=name, prior=prior, focal_px=focal, seed=seed,
                            version=version, mode=mode, slot=slot, ok=out['ok'], reason=out['reason'],
                            agreement=agreement(Ha, out['H'], w, h),
                            naive=agreement(Ha, naive, w, h), H=out['H'].tolist(), adobe_H=Ha.tolist(),
                            source_focal_px=out['source_focal_px'],
                            rotation_deg=np.degrees(out['rotation']).tolist() if out['rotation'] else None,
                            rotation_damping=out['damping'],
                            detected_keystone_deg=float(np.degrees(out['vertical_keystone_rad']))
                                if out.get('vertical_keystone_rad') is not None else None,
                            port_world_axis_deg=deviation(out['H'], truth, np.array([0,0,1,1]), w, h),
                            adobe_world_axis_deg=deviation(Ha, truth, np.array([0,0,1,1]), w, h),
                            port_horizon_error_deg=horizon_error(out['H'], pose, w, h),
                            adobe_horizon_error_deg=horizon_error(Ha, pose, w, h),
                            port_visible_convergence_deg=convergence(out['H'], verticals, w, h),
                            adobe_visible_convergence_deg=convergence(Ha, verticals, w, h))
                        row['port_removed_fraction'] = 1-row['port_visible_convergence_deg']/initial_conv
                        row['adobe_removed_fraction'] = 1-row['adobe_visible_convergence_deg']/initial_conv
                        rows.append(row)
        print(f'completed {name}', flush=True)
    summary = {}
    for prior in ('scene_focal', 'adobe_stored_focal'):
        summary[prior] = {}
        for version in ('before', 'after'):
            summary[prior][version] = {}
            for mode in ('all', *SLOTS):
                subset = [r for r in rows if r['prior'] == prior and r['version'] == version
                          and (mode == 'all' or r['mode'] == mode)]
                summary[prior][version][mode] = dict(attempts=len(subset),
                    rejected=sum(not r['ok'] for r in subset),
                    **{label: float(np.mean([r[field][metric] for r in subset]))
                       for label, field, metric in (
                           ('raw_port', 'agreement', 'raw_rms_pct'),
                           ('raw_naive', 'naive', 'raw_rms_pct'),
                           ('shape_port', 'agreement', 'shape_rms_pct'),
                           ('shape_naive', 'naive', 'shape_rms_pct'))})
    return dict(protocol='Four existing Adobe cache files; slots 1/2/3/4; all modes equally weighted; '
                'image detector; seeds paired; scene focal primary; Adobe focal sensitivity; '
                'naive = exact K(true) R(true) K(true)^-1 for every mode; no crop/zoom; '
                'ground truth only used in scoring and naive oracle; no cache row omitted',
                scenes=scenes, summary=summary, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'evidence'/'round4_adobe.json')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    result = run(range(args.seeds))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
