"""Round 7: recover output translation, then score fresh opt-in solves offline."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys

import cv2
import numpy as np
from scipy.linalg import rq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'port'), str(ROOT)]
import upright_port as up
from benchmark_adobe import agreement, mapped_points
from validate_vs_adobe import CASES, SLOTS, parse_adobe

CORNERS = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
TARGETS = dict(zip(CASES, [23.55, 25.38, 23.52, 24.08]))


def quad(H):
    q = CORNERS @ H.T
    return q[:, :2] / q[:, 2:]


def bounds_centre(q):
    s = np.sort(q, axis=0)
    inner, outer = (s[1]+s[2])/2, (s[0]+s[3])/2
    return .9*inner + .1*outer


def native_analysis():
    rows = []
    for scene, case in CASES.items():
        path = ROOT / 'adobe_truth' / f"img_{case['text']}.lua"
        text = path.read_text()
        metadata = {k: float(re.search(k + r'\s*=\s*([-\d.eE]+)', text)[1])
                    for k in ('PerspectiveScale', 'PerspectiveX', 'PerspectiveY',
                              'UprightFocalLength35mm', 'UprightCenterNormX', 'UprightCenterNormY')}
        assert metadata['PerspectiveScale'] == 100
        image = cv2.imread(str(ROOT/'lrtest/in'/f'{scene}.tif'), 0)
        h, w = image.shape
        units = np.array([w, h])
        focal = up.focal_px_from_35mm(metadata['UprightFocalLength35mm'], w, h)
        K0 = up._intrinsics_norm(focal, w, h)
        P = np.diag([w, h, 1.])
        for mode, slot in SLOTS.items():
            H = parse_adobe(path)[slot]
            # Fixed source gauge; retain the SAME native destination linear block
            # and SAME rotation. This isolates translation without fitting shape.
            K1, R = rq(P @ H @ K0)
            signs = np.diag(np.where(np.diag(K1) < 0, -1., 1.))
            K1, R = K1 @ signs, signs @ R
            assert np.linalg.det(R) > 0
            K1 = K1 / K1[2, 2]
            Kc = K1.copy(); Kc[:2, 2] = units/2
            centred = np.linalg.inv(P) @ Kc @ R @ np.linalg.inv(K0)
            centred /= centred[2, 2]
            shift = (K1[:2, 2] - units/2)/units
            T = np.eye(3); T[:2, 2] = shift
            np.testing.assert_allclose(T @ centred, H, atol=2e-15, rtol=0)
            predicted = up.frame_output(centred)
            np.testing.assert_allclose(predicted, H, atol=2e-9, rtol=0)
            # Report the source-centre gauge sensitivity separately from framing.
            stored_K0 = K0.copy()
            stored_K0[:2, 2] = [metadata['UprightCenterNormX'], metadata['UprightCenterNormY']]
            stored_K1, stored_R = rq(P @ H @ stored_K0)
            signs = np.diag(np.where(np.diag(stored_K1) < 0, -1., 1.))
            stored_K1, stored_R = stored_K1 @ signs, signs @ stored_R
            stored_K1 /= stored_K1[2, 2]
            stored_c = stored_K1.copy(); stored_c[:2, 2] = units/2
            stored_centred = np.linalg.inv(P) @ stored_c @ stored_R @ np.linalg.inv(stored_K0)
            np.testing.assert_allclose(up.frame_output(stored_centred), H, atol=2e-9, rtol=0)
            q = quad(centred)
            centre_point = centred @ [.5, .5, 1.]
            centre_point = centre_point[:2]/centre_point[2]
            hypotheses = {
                'bbox_midpoint': .5-(q.min(0)+q.max(0))/2,
                'source_centre_anchor': .5-centre_point,
                'corner_mean': .5-q.mean(0),
                'crop_origin_min_to_zero': -q.min(0),
                'grid_mean': .5-mapped_points(centred, w, h).mean(0)/units,
                'inner90_outer10': .5-bounds_centre(q),
            }
            native_q = quad(H)
            s = np.sort(native_q, axis=0)
            inner, outer = (s[1]+s[2])/2, (s[0]+s[3])/2
            alpha = np.divide(.5-inner, outer-inner,
                out=np.full(2, np.nan), where=np.abs(outer-inner) > 1e-7)
            rows.append(dict(scene=scene, mode=mode, slot=slot, metadata=metadata,
                H=H.tolist(), centred_H=centred.tolist(),
                stored_origin_translation=H[:2, 2].tolist(),
                centred_origin_translation=centred[:2, 2].tolist(),
                output_shift_norm=shift.tolist(), output_shift_px=(shift*units).tolist(),
                native_bbox_mid_norm=((native_q.min(0)+native_q.max(0))/2).tolist(),
                native_weighted_centre_norm=bounds_centre(native_q).tolist(),
                inferred_outer_weight=[float(x) if np.isfinite(x) else None for x in alpha],
                hypothesis_shift_error_px={k: float(np.linalg.norm((v-shift)*units))
                                           for k, v in hypotheses.items()},
                native_f1_xy_px=np.diag(K1)[:2].tolist(), stored_focal_px=focal,
                f1_geomean_over_f0=float(np.sqrt(K1[0, 0]*K1[1, 1])/focal),
                destination_aspect=float(K1[0, 0]/K1[1, 1]),
                destination_skew_over_fx=float(K1[0, 1]/K1[0, 0]),
                stored_source_centre_shift_px=(stored_K1[:2, 2]-units/2).tolist(),
                stored_source_centre_scale=float(np.sqrt(stored_K1[0, 0]*stored_K1[1, 1])/focal)))
    # Independent support from Round 6's existing read-only native snapshot.
    cache = json.loads((ROOT/'evidence/round6_native_slots.json').read_text())
    sweep = []
    for row in cache['rows']:
        for slot in (1, 2, 3, 4):
            H = np.asarray(row['slots'][str(slot)]).reshape(3, 3)
            error = float(np.max(np.abs(up.frame_output(H)-H/H[2, 2])))
            sweep.append(dict(name=row['name'], slot=slot, matrix_error=error))
    return rows, sweep


def measure():
    prior = json.loads((ROOT/'evidence/round6_adobe.json').read_text())
    baseline = {(r['scene'], r['prior'], r['seed'], r['mode']): r
                for r in prior['rows'] if r['version'] == 'after'}
    rows = []
    for scene, case in CASES.items():
        image = cv2.imread(str(ROOT/'lrtest/in'/f'{scene}.tif'), 0)
        h, w = image.shape
        segments = up.detect_segments(image)
        slots = parse_adobe(ROOT/'adobe_truth'/f"img_{case['text']}.lua")
        f35 = prior['scenes'][scene]['adobe_metadata']['UprightFocalLength35mm']
        for focal_name, focal in (('scene_focal', 1600.), ('adobe_stored_focal',
                                 up.focal_px_from_35mm(f35, w, h))):
            for seed in range(5):
                for mode, slot in SLOTS.items():
                    original = up.solve_ex(segments, w, h, focal, mode, seed=seed)
                    framed = up.solve_ex(segments, w, h, focal, mode, seed=seed, framing=True)
                    cached = baseline[(scene, focal_name, seed, mode)]
                    np.testing.assert_array_equal(original['H'], cached['H'])
                    assert original['ok'] == cached['ok'] and framed['ok']
                    np.testing.assert_array_equal(framed['H_unframed'], original['H'])
                    np.testing.assert_array_equal(framed['H'], up.frame_output(original['H']))
                    for key in ('rotation', 'damping', 'source_focal_px', 'quality'):
                        assert framed[key] == original[key]
                    before = agreement(slots[slot], original['H'], w, h)
                    after = agreement(slots[slot], framed['H'], w, h)
                    np.testing.assert_allclose(after['shape_rms_pct'], before['shape_rms_pct'],
                                               atol=1e-12, rtol=0)
                    assert after['raw_rms_pct'] <= TARGETS[scene]
                    a, b = mapped_points(slots[slot], w, h), mapped_points(framed['H'], w, h)
                    translation_only = np.sqrt(np.mean(np.sum(
                        ((a-a.mean(0))-(b-b.mean(0)))**2, axis=1)))/np.hypot(w, h)*100
                    rows.append(dict(scene=scene, prior=focal_name, mode=mode, slot=slot,
                        seed=seed, ok=framed['ok'], H=framed['H'].tolist(),
                        H_unframed=original['H'].tolist(), before=before, after=after,
                        optimal_translation_residual_pct=float(translation_only)))
        print('measured', scene, flush=True)
    return rows


def mean(rows, version, metric):
    return float(np.mean([r[version][metric] for r in rows]))


def write_report(result):
    native, rows = result['native'], result['rows']
    lines = [
        'ROUND 7 — OUTPUT FRAMING — 2026-10-28', '='*110,
        'RESULT: Translation framing closes most of the raw gap, without any damping or geometry change.',
        'Every framed scene/mode/seed passes the supplied scene raw target; shape is unchanged to 1e-12.',
        'This recovers a precise output TRANSLATION policy. The complete destination scale policy is not recovered.',
        '', '1. DERIVATION FROM STORED MATRICES',
        'For each output axis, sort the four transformed source-corner coordinates: s0 <= s1 <= s2 <= s3.',
        'Outer bounds = [s0,s3]; inner bounds = [s1,s2]. The latter are order statistics, not an inscribed crop.',
        'Blend bounds = [.1*s0+.9*s1, .9*s2+.1*s3]. Adobe centres their midpoint at .5:',
        '  c = (s0 + 9*s1 + 9*s2 + s3)/20; translation = (.5,.5)-c; H_framed = T @ H.',
        'The 10% outer weight is inferred independently from (.5-inner_mid)/(outer_mid-inner_mid).',
        'Where a component is symmetric the weight is unidentifiable there, but the translation remains zero.',
        'All 16 active adobe_truth slots already satisfy c=(.5,.5) to the 9-decimal storage precision.',
        'Slots 0 and 5 are identity in every requested scene: zero framing translation, unit scale.',
        'This is neither ordinary bbox recentering, transformed-centre anchoring, a mean-centroid rule,',
        'nor moving a crop origin to zero. Sorting per axis also accounts for roll-dependent corner order.',
        '',
        'Controlled same-rotation comparison: K0 uses stored source focal and centred source principal point,',
        'matching the port source-centre convention. Stored-source-centre sensitivity is reported separately.',
        'RQ(diag(w,h,1) H K0) = K1 R. Keep native R and K1 focal/aspect/skew; replace only K1 centre',
        'with (w/2,h/2). Then H_adobe = T H_centred exactly. The proposed rule recovers that T.',
        'A general upper-triangular K1 avoids falsely calling aspect/skew errors framing or changing R.',
        'This decomposition fixes a source gauge; it does not uniquely recover Adobe internal cameras.',
        'Values below: H[:,2] is normalized origin translation; Delta is the constant post-map translation.',
        'Source dimensions are 1400x1000. Slots: 1 Auto, 2 Full, 3 Level, 4 Vertical.',
        'scene       slot  Adobe H02,H12        centred H02,H12       Delta x,y pixels       rule error px',
    ]
    for r in native:
        a,b,d = r['stored_origin_translation'],r['centred_origin_translation'],r['output_shift_px']
        lines.append(f"{r['scene']:<11} {r['slot']:>2}  {a[0]: .8f},{a[1]: .8f}  "
                     f"{b[0]: .8f},{b[1]: .8f}  {d[0]: 10.4f},{d[1]: 10.4f}  "
                     f"{r['hypothesis_shift_error_px']['inner90_outer10']:.3g}")
    lines += ['', 'Source-gauge sensitivity: using the stored source principal point instead of (.5,.5).',
              'The SAME 10/90 rule still recovers native H to <=2e-9, with no refit of weights.',
              'scene       slot  Delta x,y pixels (stored-source-centre gauge)  destination scale proxy']
    for r in native:
        x,y=r['stored_source_centre_shift_px']
        lines.append(f"{r['scene']:<11} {r['slot']:>2} {x:14.5f},{y:14.5f} {r['stored_source_centre_scale']:24.8f}")
    lines += ['', 'Shift prediction errors per scene/slot, Euclidean pixels (no scale/shape fitting):',
              'scene       slot bbox-mid source-centre corner-mean grid-mean crop-origin inner90/outer10']
    for r in native:
        e=r['hypothesis_shift_error_px']
        lines.append(f"{r['scene']:<11} {r['slot']:>2} " + ' '.join(f'{e[k]:11.6f}' for k in
            ('bbox_midpoint','source_centre_anchor','corner_mean','grid_mean','crop_origin_min_to_zero','inner90_outer10')))
    lines += ['', 'Additional existing native sweep snapshot (not used to fit the rule):']
    sweep = result['sweep']
    failures = [r for r in sweep if r['matrix_error'] > 2e-9]
    lines.append(f"{len(sweep)-len(failures)}/{len(sweep)} slots satisfy the rule to max matrix error <=2e-9.")
    for r in failures:
        lines.append(f"  Exception retained: {r['name']} slot {r['slot']}, max matrix change {r['matrix_error']:.9f}.")
    lines += [
        'Both exceptions are V40S3 Full/Vertical; Round 6 already identifies its Full as a bad native solution.',
        'All 80 Auto and all 80 Level sweep slots satisfy the rule. No exception is omitted or fit away.',
        '', '2. SCALE: PerspectiveScale=100 IS NOT THE WHOLE DESTINATION SCALE',
        'All four photos store PerspectiveScale=100, PerspectiveX=0, PerspectiveY=0. There is no manual',
        'scale or shift to apply from those settings. But the native map still has an implicit destination',
        'focal/scale distinct from its source. Absolute focal and a post-uniform zoom are interchangeable',
        'in K1, so these matrices alone cannot identify an internal focal-vs-framing split.',
        'The exact translation law needs no output scale fit. We therefore ship that measured law with',
        'identity additional scale, and do not claim that native destination scale is generally identity.',
        'The stored-gauge RQ values show this directly (fx/fy/skew are NOT all forced to square pixels):',
        'scene       slot  source f px   dest fx px  dest fy px  sqrt(fx*fy)/f0  fx/fy   skew/fx',
    ]
    for r in native:
        fx,fy=r['native_f1_xy_px']
        lines.append(f"{r['scene']:<11} {r['slot']:>2} {r['stored_focal_px']:12.4f} {fx:11.4f} {fy:11.4f} "
            f"{r['f1_geomean_over_f0']:16.8f} {r['destination_aspect']:8.5f} {r['destination_skew_over_fx']:10.6f}")
    lines += [
        '', '3. IMPLEMENTATION AND SCOPE',
        'port/upright_port.py: public frame_output(H), plus solve/solve_ex(..., framing=True).',
        'Default framing=False preserves every unframed output. Opt-in solve_ex retains H_unframed.',
        'Framing runs after the original solve, gate and any explicit damping override; geometry core,',
        'detector, focal policy, all response constants and schedules remain unchanged. No native matrix',
        'or scene identifier is read by the solver. The helper uses only H and the unit source rectangle.',
        'It validates finite 3x3 input/nonzero H22 and rejects projective horizons across the source frame.',
        'Off and rejected solves remain identity. Documentation and tests cover automatic/guided modes,',
        'overrides, invariance, invalid input, off/rejection, and all 16 cached active native slots.',
        f"AST comparison: all {result['unchanged_functions']} pre-existing functions except solve_ex/solve are unchanged.",
        '', '4. FRESH MEASUREMENTS — ALL SCENES, MODES, BOTH FOCAL PROTOCOLS, FIVE SEEDS',
        'Raw/shape = fixed 25x25 source grid, percent SOURCE diagonal, same Round 4/6 metric.',
        'Shape removes positive uniform scale and translation only; it retains rotation/shear/perspective.',
        'Fresh default solves exactly reproduce all 160 Round 6 production matrices and decisions.',
        'Each opt-in solve is recomputed; H_unframed equals its paired default bit-for-bit, with identical',
        'rotation, damping, quality and source focal. All 160 attempts accepted; no identity/failure omitted.',
        'Each individual framed raw is <= its scene target, not just the following five-seed mean.',
        'START targets (Auto): A=23.55, B=25.38, C=23.52, D=24.08. Current unframed is the Round 6 baseline.',
        'prior               scene       mode      raw before raw framed shape before shape framed scale*  residual T*',
    ]
    for prior in ('scene_focal','adobe_stored_focal'):
        for scene in CASES:
            for mode in SLOTS:
                subset=[r for r in rows if (r['prior'],r['scene'],r['mode']) == (prior,scene,mode)]
                lines.append(f"{prior:<19} {scene:<11} {mode:<8} "
                    f"{mean(subset,'before','raw_rms_pct'):10.6f} {mean(subset,'after','raw_rms_pct'):10.6f} "
                    f"{mean(subset,'before','shape_rms_pct'):12.6f} {mean(subset,'after','shape_rms_pct'):12.6f} "
                    f"{mean(subset,'after','fitted_scale'):7.4f} "
                    f"{np.mean([r['optimal_translation_residual_pct'] for r in subset]):11.6f}")
    lines += [
        'scale* = OFFLINE best uniform scale matching the framed port to Adobe; not used at runtime.',
        'residual T* = OFFLINE best translation-only raw lower bound; no scale is removed there.',
        '', '5. WHAT FRAMING EXPLAINS, AND THE HONEST REMAINING CLONE DISTANCE',
        'The raw framed column is the remaining returned-map clone distance, including scale and geometry.',
        'The shape column is its similarity-invariant geometry component; it is not the full clone distance.',
        'Full/Vertical retain substantial scale error despite small shape error; the best-fit scale is far',
        'from 1 on several slots. Therefore translation does NOT explain every raw point or make an exact clone.',
        'No scale is tuned against Adobe to improve the runtime scores.',
        '', 'Auto, scene focal: raw before -> remaining raw, reduction in RMS / mean squared displacement:',
    ]
    for scene in CASES:
        subset=[r for r in rows if (r['prior'],r['scene'],r['mode']) == ('scene_focal',scene,'auto')]
        b,a=mean(subset,'before','raw_rms_pct'),mean(subset,'after','raw_rms_pct')
        mse_before=np.mean([r['before']['raw_rms_pct']**2 for r in subset])
        mse_after=np.mean([r['after']['raw_rms_pct']**2 for r in subset])
        lines.append(f'  {scene:<11} {b:.6f} -> {a:.6f}; {100*(1-a/b):.3f}% RMS / {100*(1-mse_after/mse_before):.3f}% squared reduction.')
    for prior in ('scene_focal','adobe_stored_focal'):
        for mode in ('all',*SLOTS):
            subset=[r for r in rows if r['prior']==prior and (mode=='all' or r['mode']==mode)]
            lines.append(f"  Aggregate {prior:<19} {mode:<8} raw {mean(subset,'before','raw_rms_pct'):.6f} -> "
                         f"{mean(subset,'after','raw_rms_pct'):.6f}; shape {mean(subset,'after','shape_rms_pct'):.6f}.")
    lines += [
        '', '6. VALIDATION AND REPRODUCTION',
        'python3.12 evidence/round7_probe.py > evidence/round7_probe_run.txt',
        'python3.12 -m pytest -q > evidence/round7_pytest.txt 2>&1',
        'Re-run the probe after pytest to copy its final result into this report.',
        'round7.json contains all native decompositions/hypothesis errors, 320 sweep slot checks and all',
        '160 paired fresh scene/focal/seed/mode results with H, H_unframed, raw, shape and scale diagnostics.',
        'SciPy is used only for offline RQ analysis; the runtime adds no dependency.',
        f"Production SHA256: {result['production_sha256']}",
    ]
    testpath=ROOT/'evidence/round7_pytest.txt'
    lines += testpath.read_text().splitlines()[-4:] if testpath.exists() else ['Full pytest: pending.']
    lines += [
        '', 'Limits: related synthetic scenes; no new native execution, catalog writes, GUI changes or',
        'real-photo validation. The 10/90 rule is empirically exact on the requested 16 slots and 318/320',
        'sweep controls, not a claim about Adobe source or every possible failure branch. The complete',
        'destination scale policy and the existing coupled-rotation/Auto branch errors remain unresolved.',
    ]
    (ROOT/'evidence/round7.txt').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    original = ast.parse((ROOT/'evidence/round5_original_port.py').read_text())
    current = ast.parse((ROOT/'port/upright_port.py').read_text())
    functions=lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                            if isinstance(n, ast.FunctionDef)}
    old,new=functions(original),functions(current)
    unchanged=[k for k in old if k not in ('solve_ex','solve')]
    assert all(old[k]==new[k] for k in unchanged)
    globals_dump=lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body
                               if not isinstance(n, ast.FunctionDef)]
    assert globals_dump(original) == globals_dump(current)
    native,sweep=native_analysis()
    result=dict(native=native, sweep=sweep, rows=measure(), unchanged_functions=len(unchanged),
        production_sha256=hashlib.sha256((ROOT/'port/upright_port.py').read_bytes()).hexdigest())
    (ROOT/'evidence/round7.json').write_text(json.dumps(result, indent=2)+'\n')
    write_report(result)
    print('wrote evidence/round7.txt and evidence/round7.json')
