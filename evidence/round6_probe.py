"""Round 6 offline causal controls. Native transforms enter scoring only.

Uses the frozen Round 5 solver and the read-only slot snapshot, never Lightroom.
All assertions validate the diagnostic geometry; production is not modified.
"""
from pathlib import Path
import importlib.util
import json
import math
import sys
from unittest.mock import patch

import cv2
import numpy as np
from scipy.linalg import rq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'port'), str(ROOT)]
from benchmark_adobe import agreement, mapped_points, convergence, visible_verticals
from validate_vs_adobe import CASES, parse_adobe, to_pix
from round5_limits import ideal_fraction
import analyze_sigma_final as sweep
import make_test_images as M

spec = importlib.util.spec_from_file_location('round6_frozen', ROOT / 'evidence/round5_original_port.py')
up = importlib.util.module_from_spec(spec)
spec.loader.exec_module(up)
ORIGINAL_SCALE = up._auto_pitch_scale
LABELS = ('keystone13_physical', 'keystone13_euler', 'camera22_physical',
          'camera22_euler', 'camera13_euler', 'zero_pitch')


def candidate(label):
    def scale(rotation, fractions, K1, Ki, sv, w, h, keystone):
        if label == 'keystone13_physical':
            return ORIGINAL_SCALE(rotation, fractions, K1, Ki, sv, w, h, keystone)
        if label == 'keystone13_euler':
            return fractions[0]
        if label == 'zero_pitch':
            return 0.
        sigma = math.radians(13.33 if label == 'camera13_euler' else 22.26)
        fraction = up._saturate(2.64 * math.exp(-rotation[0]**2 / (2*sigma**2)))
        if label.endswith('euler'):
            return fraction
        return ORIGINAL_SCALE(rotation, (fraction, *fractions[1:]), K1, Ki, sv, w, h, keystone)
    return scale


def directions(H, segments, w, h):
    lines = up._homogeneous_lines(segments) @ np.linalg.inv(H)
    return np.arctan2(-lines[:, 1]*w, lines[:, 0]*h)


def extended_score(adobe, H, w, h, verticals):
    score = agreement(adobe, H, w, h)
    a, b = mapped_points(adobe, w, h), mapped_points(H, w, h)
    ac, bc = a-a.mean(0), b-b.mean(0)
    rho = np.sum(ac*bc)/np.sqrt(np.sum(ac*ac)*np.sum(bc*bc))
    # This residual is unchanged if either map is translated/uniformly scaled,
    # and is symmetric under exchanging Adobe and the candidate.
    score['symmetric_shape_pct'] = float(100*np.sqrt(max(0., 1-rho*rho)))
    score['translation_only_rms_pct'] = float(np.sqrt(np.mean(np.sum((ac-bc)**2, axis=1)))/np.hypot(w, h)*100)
    da = directions(adobe, verticals, w, h)
    db = directions(H, verticals, w, h)
    delta = (db-da+np.pi/2) % np.pi - np.pi/2
    score['vertical_direction_rms_deg'] = float(np.degrees(np.sqrt(np.mean(delta*delta))))
    for label, points in (('interior', np.array([(x, y, 1.) for y in np.linspace(.2, .8, 13) for x in np.linspace(.2, .8, 13)])),):
        def mapped(matrix):
            q = points @ matrix.T
            return q[:, :2]/q[:, 2:] * [w, h]
        aa, bb = mapped(adobe), mapped(H)
        aa -= aa.mean(0); bb -= bb.mean(0)
        s = np.sum(aa*bb)/np.sum(bb*bb)
        score[label+'_shape_rms_pct'] = float(np.sqrt(np.mean(np.sum((aa-s*bb)**2, axis=1)))/np.hypot(w, h)*100)
    # Generic centre anchoring is an offline control, not Adobe framing.
    q = H @ np.array([.5, .5, 1.]); T = np.eye(3); T[:2, 2] = .5-q[:2]/q[2]
    anchored = agreement(adobe, T@H, w, h)
    np.testing.assert_allclose(anchored['shape_rms_pct'], score['shape_rms_pct'], atol=1e-12, rtol=0)
    score['centre_anchored_raw_rms_pct'] = anchored['raw_rms_pct']
    # Check the symmetric statistic independently against reverse alignment.
    reverse = agreement(H, adobe, w, h)
    reverse_relative = reverse['shape_rms_px']/np.sqrt(np.mean(np.sum(bc*bc, axis=1)))*100
    np.testing.assert_allclose(reverse_relative, score['symmetric_shape_pct'], atol=1e-9, rtol=0)
    return score


def split(H, focal, w, h, centre=(.5, .5)):
    """Exact RQ with FIXED source camera; destination K is general triangular.

    This is not polar(K^-1 H K), which mixes destination framing into rotation.
    Destination skew/aspect are reported; no square-pixel exactness is assumed.
    """
    K0 = up._intrinsics_norm(focal, w, h); K0[:2, 2] = centre
    P = np.diag([w, h, 1.])
    K1, R = rq(P @ H @ K0)
    signs = np.diag(np.where(np.diag(K1) < 0, -1., 1.))
    K1, R = K1@signs, signs@R
    assert np.linalg.det(R) > 0
    rebuilt = np.linalg.inv(P) @ K1 @ R @ np.linalg.inv(K0)
    np.testing.assert_allclose(rebuilt, H, atol=1e-12, rtol=0)
    np.testing.assert_allclose(up._rotation(*up._euler_xyz(R)), R, atol=1e-12, rtol=0)
    return dict(euler_deg=np.degrees(up._euler_xyz(R)).tolist(),
                aspect=float(K1[0, 0]/K1[1, 1]), skew_over_f=float(K1[0, 1]/K1[0, 0]),
                shared_camera_polar_euler_deg=np.degrees(up._euler_xyz(
                    up._polar(np.linalg.inv(K0) @ H @ K0))).tolist())


def native_record(name, pose, slots, metadata):
    w, h = 1400, 1000
    verticals = visible_verticals(pose, 1600, w, h)
    conv0 = convergence(np.eye(3), verticals, w, h)
    result = dict(name=name, pose_deg=np.degrees(pose).tolist(), metadata=metadata, slots={})
    focal = up.focal_px_from_35mm(metadata['UprightFocalLength35mm'], w, h)
    centre = (metadata['UprightCenterNormX'], metadata['UprightCenterNormY'])
    for slot, H in slots.items():
        if int(slot) not in (1, 2, 3, 4):
            continue
        H = np.asarray(H).reshape(3, 3); H = H/H[2, 2]
        score = agreement(H, np.eye(3), w, h)
        q = mapped_points(H, w, h)-mapped_points(np.eye(3), w, h)
        result['slots'][str(slot)] = dict(H=H.tolist(), frobenius_H_minus_I=float(np.linalg.norm(H-np.eye(3))),
            max_abs_H_minus_I=float(np.max(np.abs(H-np.eye(3)))), identity_raw_rms_pct=score['raw_rms_pct'],
            identity_shape_rms_pct=score['shape_rms_pct'], max_displacement_px=float(np.linalg.norm(q, axis=1).max()),
            physical_removed=float(1-convergence(H, verticals, w, h)/conv0),
            true_K0=split(H, 1600, w, h),
            stored_focal_centred_K0=split(H, focal, w, h),
            stored_K0=split(H, focal, w, h, centre))
    return result


def native_analysis():
    baseline = json.loads((ROOT/'evidence/round5_adobe.json').read_text())
    records = []
    for name, c in CASES.items():
        records.append(native_record(name, [c[k] for k in ('rx', 'ry', 'rz')],
            parse_adobe(ROOT/'adobe_truth'/f"img_{c['text']}.lua"), baseline['scenes'][name]['adobe_metadata']))
    cache = json.loads((ROOT/'evidence/round6_native_slots.json').read_text())
    old = {r['name']: r for r in json.loads((ROOT/'evidence/sigma_v_precise.json').read_text())['rows']}
    measured = []
    for row in cache['rows']:
        name = row['name']; angle = float(name[1:3]); pose = [-math.radians(angle), 0., 0.]
        record = native_record(name, pose, row['slots'], row['metadata'])
        records.append(record)
        H = np.array(row['slots']['1']).reshape(3, 3)
        init, after = sweep.measure(math.radians(angle), to_pix(H, 1400, 1000))
        keep = np.isfinite(after)
        fraction = float(1-after[keep].mean()/init[keep].mean())
        if name in old:
            np.testing.assert_allclose(fraction, old[name]['frac'], atol=1e-12, rtol=0)
        eulers = {g: record['slots']['1'][g]['euler_deg'][0]/record['slots']['2'][g]['euler_deg'][0]
                  for g in ('true_K0', 'stored_focal_centred_K0', 'stored_K0')}
        measured.append(dict(name=name, angle=angle, old76=name in old,
            physical_fraction=fraction, physical_convergence_fraction=record['slots']['1']['physical_removed'],
            euler_fractions=eulers,
            full_pitch_deg=abs(record['slots']['2']['true_K0']['euler_deg'][0])))
    fits = {}
    for subset in ('original76', 'current80'):
        rows = [r for r in measured if r['old76'] or subset == 'current80']
        fits[subset] = {}
        for target in ('physical_fraction', 'physical_convergence_fraction', 'euler_true_K0', 'euler_stored_K0'):
            t = np.array([r['angle'] for r in rows])
            f = np.array([r['euler_fractions'][target[6:]] if target.startswith('euler_') else r[target] for r in rows])
            L0, sigma = sweep.fit(t, f)
            diagnostics = {}
            for value in (13.33, 22.26):
                q = sweep.model(np.log([2.64, value]), t)
                diagnostics[str(value)] = dict(rms=float(np.sqrt(np.mean((q-f)**2))), max_abs=float(np.max(np.abs(q-f))))
            fitted = sweep.model(np.log([L0, sigma]), t)
            fits[subset][target] = dict(L0=L0, sigma_deg=sigma, rms=float(np.sqrt(np.mean((fitted-f)**2))), fixed_curves=diagnostics)
    return dict(records=records, measured=measured, fits=fits,
                cache_additional_to_original76=[r['name'] for r in measured if not r['old76']])


def candidate_analysis():
    baseline = json.loads((ROOT/'evidence/round5_adobe.json').read_text())
    rows = []
    for name, case in CASES.items():
        adobe = parse_adobe(ROOT/'adobe_truth'/f"img_{case['text']}.lua")[1]
        image = cv2.imread(str(ROOT/'lrtest/in'/(name+'.tif')), 0)
        h, w = image.shape; segments = up.detect_segments(image)
        pose = [case[k] for k in ('rx', 'ry', 'rz')]
        verticals = visible_verticals(pose, 1600, w, h)
        conv0 = convergence(np.eye(3), verticals, w, h)
        f35 = baseline['scenes'][name]['adobe_metadata']['UprightFocalLength35mm']
        for prior, focal in (('scene_focal', 1600.), ('adobe_stored_focal', up.focal_px_from_35mm(f35, w, h))):
            for seed in range(5):
                for label in LABELS:
                    with patch.object(up, '_auto_pitch_scale', candidate(label)):
                        out = up.solve_ex(segments, w, h, focal, 'auto', seed=seed)
                    rows.append(dict(scene=name, prior=prior, seed=seed, variant=label,
                        ok=out['ok'], reason=out['reason'], H=out['H'].tolist(),
                        rotation_deg=np.degrees(out['rotation']).tolist(), damping=out['damping'],
                        detected_keystone_deg=float(np.degrees(out['vertical_keystone_rad'])),
                        removed=1-convergence(out['H'], verticals, w, h)/conv0,
                        **extended_score(adobe, out['H'], w, h, verticals)))
        print('candidates', name, flush=True)
    mild = []
    for name, folder in (('NM_single_wall', 'nm'), ('NM_railing', 'nm'), ('NM_curves', 'nm'), ('NM_three_plane', 'nm'), ('CT_manhattan2', 'nm2')):
        image = cv2.imread(str(ROOT/'lrtest'/folder/(name+'.tif')), 0)
        h, w = image.shape; segments = up.detect_segments(image)
        for seed in range(5):
            for label in LABELS:
                with patch.object(up, '_auto_pitch_scale', candidate(label)):
                    out = up.solve_ex(segments, w, h, 1600, 'auto', seed=seed)
                mild.append(dict(scene=name, seed=seed, variant=label, ok=out['ok'],
                    ideal_fraction=ideal_fraction(out['H'], [M.PITCH15, M.YAW6, 0.])))
    return dict(rows=rows, mild=mild)


if __name__ == '__main__':
    result = dict(native=native_analysis(), candidates=candidate_analysis())
    (ROOT/'evidence/round6_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print('native fits', json.dumps(result['native']['fits'], indent=2))
