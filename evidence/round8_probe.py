"""Offline destination-scale identification. Cached Adobe maps are evidence only."""
from pathlib import Path
import ast
import hashlib
import json
import sys

import cv2
import numpy as np
from scipy.linalg import rq
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'port'), str(ROOT)]
import upright_port as up
from benchmark_adobe import agreement, mapped_points

WH = np.array([1400., 1000.])
C = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])


def quad(H):
    p = C @ H.T
    if np.min(p[:, 2]) <= 0:
        raise ValueError('horizon')
    return p[:, :2]/p[:, 2:]


def area(q):
    return float(abs(np.sum(q[:, 0]*np.roll(q[:, 1], -1)-q[:, 1]*np.roll(q[:, 0], -1)))/2)


def blend(q):
    b = np.sort(q, axis=0)
    return .1*b[0]+.9*b[1], .9*b[2]+.1*b[3]


def zoom(H, s):
    return np.array([[s, 0, (1-s)/2], [0, s, (1-s)/2], [0, 0, 1.]]) @ H


def cover_scale(q):
    # Half-plane test for all four original-frame corners inside s*(q-.5)+.5.
    p = q-.5
    e = np.roll(p, -1, axis=0)-p
    normals = np.column_stack([e[:, 1], -e[:, 0]])
    d = np.sum(normals*p, axis=1)
    if np.any(d <= 0):
        return float('nan')
    return float(np.max(np.sum(abs(normals), axis=1)/2/d))


def retained(q):
    # Exact convex clipping against the unit output frame; retained OUTPUT area.
    p = list(q)
    for axis, bound, sign in ((0, 0., 1), (0, 1., -1), (1, 0., 1), (1, 1., -1)):
        out = []
        for a, b in zip(p, p[1:]+p[:1]):
            ia, ib = sign*(a[axis]-bound) >= 0, sign*(b[axis]-bound) >= 0
            if ia:
                out.append(a)
            if ia != ib:
                t = (bound-a[axis])/(b[axis]-a[axis])
                out.append(a+t*(b-a))
        p = out
        if not p:
            return 0.
    return area(np.array(p))/area(q)


def features(H):
    q = quad(H)
    lo, hi = blend(q)
    return dict(quad_area=area(q), blend_area=float(np.prod(hi-lo)),
                retained_fraction=retained(q),
                corner_displacement=float(np.max(np.linalg.norm((q-C[:, :2])*WH, axis=1))/np.linalg.norm(WH)),
                contain_scale=float(1/(2*np.max(abs(q-.5)))), cover_scale=cover_scale(q),
                inner_fit=float(1/np.max(np.diff(np.sort(q, axis=0)[1:3], axis=0))),
                blend_fit=float(1/np.max(hi-lo)))


def decompose(name, mode, slot, H, metadata, group):
    f = up.focal_px_from_35mm(metadata['UprightFocalLength35mm'], *WH)
    K0 = up._intrinsics_norm(f, *WH)
    P = np.diag([*WH, 1.])
    K1, R = rq(P @ H @ K0)
    signs = np.diag(np.where(np.diag(K1) < 0, -1., 1.))
    K1, R = K1@signs, signs@R
    K1 /= K1[2, 2]
    s = float(np.sqrt(K1[0, 0]*K1[1, 1])/f)
    B = K1.copy()
    B[:2, :2] /= s
    B[:2, 2] = WH/2
    base = np.linalg.inv(P) @ B @ R @ np.linalg.inv(K0)
    base /= base[2, 2]
    framed = up.frame_output(base)
    native = up.frame_output(zoom(framed, s))
    error = float(np.max(abs(native-H/H[2, 2])))
    isotropic = K0 @ R @ np.linalg.inv(K0)
    isotropic /= isotropic[2, 2]
    isotropic = up.frame_output(isotropic)
    a, b = mapped_points(H, *WH), mapped_points(framed, *WH)
    centre = WH/2
    oracle = float(np.sum((a-centre)*(b-centre))/np.sum((b-centre)**2))
    assert abs(oracle-s) < 2e-8 or error > 2e-9
    before, after = features(framed), features(H)
    angles = np.degrees(up._euler_xyz(R))
    correction = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
    return dict(name=name, group=group, mode=mode, slot=slot, scale=s,
                pitch=float(angles[0]), yaw=float(angles[1]), roll=float(angles[2]),
                abs_pitch=float(abs(angles[0])), abs_yaw=float(abs(angles[1])), abs_roll=float(abs(angles[2])),
                correction_deg=float(correction),
                correction_rms=float(np.sqrt(np.mean(np.sum((b-mapped_points(np.eye(3), *WH))**2, axis=1)))/np.linalg.norm(WH)),
                before=before, after=after, base=base.tolist(), framed=framed.tolist(),
                H=H.tolist(), reconstruction_error=error,
                isotropic_scale=agreement(H, isotropic, *WH)['fitted_scale'],
                isotropic_shape=agreement(H, isotropic, *WH)['shape_rms_pct'])


def dataset():
    native = json.loads((ROOT/'evidence/round7.json').read_text())['native']
    rows = [decompose(r['scene'], r['mode'], r['slot'], np.array(r['H']), r['metadata'], 'four_scene') for r in native]
    cache = json.loads((ROOT/'evidence/round6_native_slots.json').read_text())
    modes = dict(zip((1, 2, 3, 4), ('auto', 'full', 'level', 'vertical')))
    for r in cache['rows']:
        for slot, mode in modes.items():
            rows.append(decompose(r['name'], mode, slot, np.array(r['slots'][str(slot)]).reshape(3, 3), r['metadata'], 'sweep'))
    return rows


def correlations(rows):
    fields = ['pitch', 'yaw', 'roll', 'abs_pitch', 'abs_yaw', 'abs_roll', 'correction_deg', 'correction_rms']
    values = {k: [r[k] for r in rows] for k in fields}
    values.update({f'{stage}_{k}': [r[stage][k] for r in rows]
                   for stage in ('before', 'after')
                   for k in ('quad_area', 'blend_area', 'retained_fraction', 'corner_displacement')})
    values.update({f'mode_{m}': [float(r['mode']==m) for r in rows] for m in ('auto', 'full', 'level', 'vertical')})
    y = [r['scale'] for r in rows]
    return {k: dict(pearson=float(pearsonr(x, y).statistic), spearman=float(spearmanr(x, y).statistic))
            for k, x in values.items() if np.ptp(x) > 1e-12 and np.std(x)>1e-9}


def policies(r):
    f = r['before']
    H = np.array(r['framed'])
    invq = quad(np.linalg.inv(H))
    return dict(identity=1., preserve_quad_area=1/np.sqrt(f['quad_area']),
                unit_blend_area=1/np.sqrt(f['blend_area']),
                fit_contain=f['contain_scale'], fit_cover=f['cover_scale'],
                fit_inner=f['inner_fit'], fit_blend=f['blend_fit'],
                inverse_bbox=float(np.max(np.ptp(invq, axis=0))))


def summary_predictions(rows, predictions):
    errs = np.array(predictions)-[r['scale'] for r in rows]
    raw = [agreement(r['H'], zoom(np.array(r['framed']), s), *WH)['raw_rms_pct'] for r, s in zip(rows, predictions)]
    return dict(scale_rmse=float(np.sqrt(np.mean(errs**2))), max_scale_error=float(np.max(abs(errs))),
                mean_raw=float(np.mean(raw)), max_raw=float(np.max(raw)))


def hypotheses(rows):
    out = {k: summary_predictions(rows, [policies(r)[k] for r in rows]) for k in policies(rows[0])}
    # Learned constants validated with all slots of the test scene held out.
    for kind, per_mode in [(k, m) for k in ('quad_area', 'blend_area', 'retained_fraction', 'corner_displacement') for m in (False, True)]:
        pred = []
        for r in rows:
            train = [a for a in rows if a['name'] != r['name'] and (not per_mode or a['mode']==r['mode'])]
            target = float(np.mean([a['after'][kind] for a in train]))
            if kind.endswith('area'):
                s = np.sqrt(target/r['before'][kind])
            else:
                q = quad(np.array(r['framed']))
                def value(s):
                    qq = .5+s*(q-.5)
                    return retained(qq) if kind == 'retained_fraction' else float(np.max(np.linalg.norm((qq-C[:, :2])*WH, axis=1))/np.linalg.norm(WH))
                if kind == 'retained_fraction':
                    s = brentq(lambda x: value(x)-target, 1e-4, 100.)
                else:
                    minimum = minimize_scalar(value, bounds=(1e-4, 10.), method='bounded')
                    if target < minimum.fun:
                        s = minimum.x
                    else:
                        roots = [brentq(lambda x: value(x)-target, minimum.x, 100.)]
                        if value(1e-4) >= target:
                            roots.append(brentq(lambda x: value(x)-target, 1e-4, minimum.x))
                        s = min(roots, key=lambda x: abs(x-1))
            pred.append(s)
        stats = summary_predictions(rows, pred)
        values = [r['after'][kind] for r in rows]
        stats.update(native_min=float(min(values)), native_max=float(max(values)), native_std=float(np.std(values)))
        out['fixed_'+kind+('_per_mode' if per_mode else '')+'_LOSO'] = stats
    return out


def recovered_rule(rows):
    for r in rows:
        H = np.array(r['framed'])
        b = np.sort(quad(H), axis=0)
        inner = .5/float(np.max(abs(b[1:3]-.5)))
        cover = r['before']['cover_scale']
        predicted = up.scale_output(H)
        s = float(np.sqrt(area(quad(predicted))/area(quad(H))))
        actual = r['scale']
        native_branch = 'cover' if abs(cover-actual)<abs(inner-actual) else 'inner'
        r['rule'] = dict(inner=inner, cover=cover, ratio=cover/inner,
                         native_branch=native_branch, predicted_branch='cover' if cover<=1.1*inner else 'inner',
                         predicted_scale=s, scale_error=s-actual,
                         raw=agreement(r['H'], predicted, *WH)['raw_rms_pct'])
        # Framing and size normalization commute, including native exceptions.
        np.testing.assert_allclose(up.frame_output(up.scale_output(np.array(r['base']))), predicted, atol=2e-14, rtol=0)


def measure():
    from validate_vs_adobe import CASES, SLOTS, parse_adobe
    cached = json.loads((ROOT/'evidence/round7.json').read_text())
    rows = []
    for scene, case in CASES.items():
        image = cv2.imread(str(ROOT/'lrtest/in'/f'{scene}.tif'), 0)
        segments = up.detect_segments(image)
        adobe = parse_adobe(ROOT/'adobe_truth'/f"img_{case['text']}.lua")
        for r in cached['rows']:
            if r['scene'] != scene:
                continue
            focal = 1600. if r['prior']=='scene_focal' else next(a['stored_focal_px'] for a in cached['native'] if a['scene']==scene)
            original = up.solve_ex(segments, *WH, focal, r['mode'], seed=r['seed'])
            final = up.solve_ex(segments, *WH, focal, r['mode'], seed=r['seed'], framing=True, scaling=True)
            assert original['ok'] and final['ok']
            np.testing.assert_array_equal(original['H'], r['H_unframed'])
            np.testing.assert_array_equal(final['H_unframed'], original['H'])
            np.testing.assert_array_equal(final['H_unscaled'], r['H'])
            np.testing.assert_array_equal(final['H'], up.scale_output(final['H_unscaled']))
            for key in ('rotation', 'damping', 'quality', 'source_focal_px'):
                assert original[key] == final[key]
            A, B = adobe[SLOTS[r['mode']]], np.array(r['H'])
            a, b = mapped_points(A, *WH), mapped_points(B, *WH)
            centre = WH/2
            # One scale, constrained pivot at framed (.5,.5), no fitted translation.
            oracle_s = float(np.sum((a-centre)*(b-centre))/np.sum((b-centre)**2))
            oracle = agreement(A, zoom(B, oracle_s), *WH)
            score = agreement(A, final['H'], *WH)
            np.testing.assert_allclose(score['shape_rms_pct'], r['after']['shape_rms_pct'], atol=1e-12, rtol=0)
            # Wrong origin-pivot order, for measuring framing interaction.
            s = float(np.sqrt(area(quad(final['H']))/area(quad(B))))
            wrong = np.diag([s, s, 1.])@B
            rows.append(dict(scene=scene, prior=r['prior'], mode=r['mode'], seed=r['seed'], ok=final['ok'],
                H=final['H'].tolist(), before=r['before'], framed=r['after'], final=score,
                applied_scale=s, oracle_centre_scale=oracle_s, centre_scale_lower_bound=oracle['raw_rms_pct'],
                shape_lower_bound=score['shape_rms_pct'], wrong_pivot_raw=agreement(A, wrong, *WH)['raw_rms_pct']))
        print('fresh paired solves', scene, flush=True)
    return rows


def report(result):
    native, measured = result['rows'], result['measurements']
    lines = ['ROUND 8 — DESTINATION SCALE — 2026-10-28', '='*115,
        'RESULT: a geometric, piecewise boundary-fit policy emerges. Implemented as a second OPT-IN step.',
        'No mode constants or damping changed. Raw clone distance is measured on fresh solves, without fitted scale.',
        'Primary mean raw: original 33.266029% -> framing 3.562255% -> framing+scale 0.587092%.',
        'Primary pooled RMS after framing+scale: 0.919553%; all aggregates and maximum errors below.',
        'Primary oracle mean lower bounds: scale alone 0.430136%; scale+translation 0.403312%.',
        '', '1. DEFINITIONS / SOURCE GAUGE / IDENTIFIABILITY',
        'Coordinates are normalized to the source rectangle; scoring is 625 fixed 25x25 grid points in pixels,',
        'RMS as percent of SOURCE diagonal (1400x1000), matching Round 7. Means below are arithmetic means of',
        'per-attempt RMS unless explicitly marked pooled RMS. All four scenes, four modes, two focal protocols,',
        'five seeds: 160 paired attempts; failures would be retained. No new native solve or catalog access.',
        'For native analysis fix K0 to stored source focal, principal point (.5,.5). RQ(P H K0)=K1 R, P=diag(w,h,1).',
        's_native=sqrt(K1.fx*K1.fy)/f0. Build unit-scale centred K1 by dividing its whole 2x2 linear block by',
        's_native and setting its centre to (w/2,h/2). Retain native aspect/skew and EXACT SAME rotation.',
        'H0=P^-1 K1_unit R K0^-1; B=frame_output(H0). H_native=S(.5,s_native) B except known framing failures.',
        'Round 7 centred_H retained native destination focal, so scale relative to THAT map is 1 by construction.',
        'This round removes that circularity by fixing its destination geometric-mean focal to f0.',
        'Separately scale*_port fits the actual centred port solution to Adobe. It includes geometry mismatch;',
        's_native is a fixed-camera-gauge measure, not a uniquely identifiable internal Adobe zoom parameter.',
        's_isotropic below uses centred K0 R K0^-1 instead; residual aspect/skew remains in shape.',
        '"Blend area" means the area of Round 7\'s 90% inner/10% outer BOUND RECTANGLE, not an inscribed quad.',
        'Quad area is the shoelace area. Before means B above; after means the stored native map.',
        'Crop fraction = area(mapped quad intersect unit frame)/area(mapped quad), in OUTPUT-area measure.',
        'This is not retained source pixels under a nonlinear Jacobian or a Lightroom crop metadata setting.',
        'A freely sized inscribed crop fraction is invariant to uniform scale and cannot identify scale.',
        '', 'scene       mode      pitch      yaw     roll  |R|deg  s_native s_isotropic scale*_port[scene/stored]',
    ]
    for r in native[:16]:
        v = [np.mean([a['framed']['fitted_scale'] for a in measured if a['scene']==r['name'] and a['mode']==r['mode'] and a['prior']==p])
             for p in ('scene_focal', 'adobe_stored_focal')]
        lines.append(f"{r['name']:<11} {r['mode']:<8} {r['pitch']:8.3f} {r['yaw']:8.3f} {r['roll']:8.3f} {r['correction_deg']:7.3f} {r['scale']:9.6f} {r['isotropic_scale']:11.6f} {v[0]:.6f}/{v[1]:.6f}")
    lines += ['', 'scene       mode      corrRMS   quad_before quad_after blend_before blend_after crop_before crop_after']
    for r in native[:16]:
        a, b = r['before'], r['after']
        lines.append(f"{r['name']:<11} {r['mode']:<8} {r['correction_rms']:8.6f} {a['quad_area']:11.6f} {b['quad_area']:10.6f} {a['blend_area']:12.6f} {b['blend_area']:11.6f} {a['retained_fraction']:11.6f} {b['retained_fraction']:10.6f}")
    lines += ['', '2. CORRELATIONS (NOT A CAUSAL FORMULA)',
        'Pearson r / Spearman rho against s_native. Mode uses one-hot/point-biserial correlations.',
        'correction_deg=acos((trace(R)-1)/2); correction_rms is translation-framed grid displacement/source diagonal.',
        'After-area correlations partly encode s^2 algebraically; they cannot predict s from an input map.',
        'The 16 scene/slot rows are related synthetic scenes, and within-mode n=4 is descriptive only.',
        'The 320 stored sweep rows have narrow yaw/roll coverage and correlated exposures; no p-value claims.',
        'No duplicate seeds are counted as new native samples. All 320 sweep slots, including bad V40S3, retained.']
    for group, modes in result['correlations'].items():
        for mode, values in modes.items():
            n = sum(r['group']==group and (mode=='all' or r['mode']==mode) for r in native)
            lines += ['', f'{group}, {mode}, n={n}: feature                              Pearson    Spearman']
            for k, v in values.items():
                lines.append(f"  {k:<45} {v['pearson']:9.5f} {v['spearman']:11.5f}")
    for prior in ('scene_focal', 'adobe_stored_focal'):
        port_rows = []
        for r in native[:16]:
            scale = np.mean([a['framed']['fitted_scale'] for a in measured
                             if (a['prior'], a['scene'], a['mode'])==(prior, r['name'], r['mode'])])
            port_rows.append(dict(r, scale=float(scale)))
        lines += ['', f'{prior}: actual port scale* vs native geometric features, n=16 (seed means, descriptive):']
        for k, v in correlations(port_rows).items():
            lines.append(f"  {k:<45} {v['pearson']:9.5f} {v['spearman']:11.5f}")
    lines += ['', '3. EXPLICIT HYPOTHESIS TESTS',
        'All scale-error units are multiplicative scale (0.01 = one scale percentage point); raw is source-diagonal %. ',
        'These controls use SAME native R/aspect/skew, so they test destination size rather than solver geometry.',
        'Identity = PerspectiveScale/100, which is 1 in all four scene metadata records.',
        'preserve_quad_area=1/sqrt(area(B)); unit_blend_area=1/sqrt(blend_area(B)).',
        'fit_contain=1/max(2*abs(q-.5)); fit_cover tests every original corner against every quad edge half-plane.',
        'fit_inner=1/max(inner widths); fit_blend=1/max(blended widths); inverse_bbox=max bbox width of B^-1.',
        'Fixed area rules fit a constant final area using all other scenes. Fixed crop fraction numerically',
        'solves retained_fraction(s)=training mean. Fixed displacement solves max corner displacement=training',
        'mean (diagonal units), taking the root nearest 1. If unreachable, use its minimum; no impossible row dropped.',
        'LOSO holds out the entire scene, including every slot; per_mode versions learn a separate constant by mode.']
    for group, hypotheses_ in result['hypotheses'].items():
        lines += ['', group+': hypothesis                                 scale RMSE max abs err  mean raw   max raw']
        for k, v in hypotheses_.items():
            lines.append(f"  {k:<49} {v['scale_rmse']:10.6f} {v['max_scale_error']:11.6f} {v['mean_raw']:9.6f} {v['max_raw']:9.6f}")
        for k in ('fixed_quad_area_LOSO', 'fixed_blend_area_LOSO', 'fixed_retained_fraction_LOSO', 'fixed_corner_displacement_LOSO'):
            v = hypotheses_[k]
            lines.append(f"  Native invariant range {k}: {v['native_min']:.8f}..{v['native_max']:.8f}; std={v['native_std']:.8f}")
    lines += ['', 'Verdicts: fixed PerspectiveScale identity FAILS; area preservation FAILS; fixed final quad/blend',
        'area FAILS, including mode-specific constants; fixed crop fraction FAILS, including mode-specific constants.',
        'A single always-active corner-displacement constant FAILS. A cap inactive below threshold also cannot',
        'explain all shrinking maps: their final max displacement is not constant. Always contain / always cover',
        'the original frame FAIL; a CONDITIONAL cover/inner-bound fit succeeds below.',
        'Specifically, the seven shrinking four-scene slots have final max corner displacement spanning',
        '0.07332467..0.27345113 source diagonals; all cannot be binding the same displacement cap.',
        '', '4. RECOVERED BOUNDARY POLICY',
        'First translate the 90/10 bounds centre c to (.5,.5). Let p_i=q_i-(.5,.5), i cyclic (TL,TR,BR,BL).',
        'For each quad edge e_i=p_(i+1)-p_i, use outward n_i=(e_iy,-e_ix), d_i=n_i dot p_i > 0.',
        's_cover = max_i (|n_ix|+|n_iy|)/(2*d_i). This is the smallest uniform scale covering the unit frame.',
        'Sort p independently per axis; keep middle two values (the inner bounds).',
        's_inner = .5 / max(abs(inner bounds)). This is the largest scale keeping that rectangle INSIDE the frame.',
        'Recovered policy: if s_cover/s_inner <= 1.10, choose s_cover; otherwise choose s_inner.',
        'Apply positive uniform scale about (.5,.5), after framing. No camera angle, mode label, scene name or',
        'native cache enters this helper. The observed mode dependence comes from quad geometry and this switch.',
        'The switch is discontinuous; the small stored corpus does not establish behavior at exact equality.',
        'The 1.10 coefficient is MEASURED, not claimed Adobe source. Both candidate boundaries were identified',
        'on the four scenes; the sweep was also inspected to identify/narrow the switch, so is NOT a blind holdout.',
        '', 'scene       mode        native s   cover s   inner s     ratio  branch  predicted s scale error   raw same-R']
    for r in native[:16]:
        v = r['rule']
        lines.append(f"{r['name']:<11} {r['mode']:<8} {r['scale']:10.7f} {v['cover']:9.6f} {v['inner']:9.6f} {v['ratio']:9.6f} {v['predicted_branch']:<6} {v['predicted_scale']:11.7f} {v['scale_error']:11.3g} {v['raw']:12.7f}")
    for group in ('four_scene', 'sweep'):
        subset = [r for r in native if r['group']==group]
        ordinary = [r for r in subset if r['reconstruction_error']<=2e-9]
        cover_ratios = [r['rule']['ratio'] for r in ordinary if r['rule']['native_branch']=='cover' and abs(r['rule']['cover']-r['rule']['inner'])>.001]
        inner_ratios = [r['rule']['ratio'] for r in ordinary if r['rule']['native_branch']=='inner' and abs(r['rule']['cover']-r['rule']['inner'])>.001]
        lines += [f"  {group}: switch interval supported by labels: [{max(cover_ratios):.9f}, {min(inner_ratios):.9f}).",
            f"  {group}: all {len(subset)} slots scale RMSE={np.sqrt(np.mean([r['rule']['scale_error']**2 for r in subset])):.9f}, max={max(abs(r['rule']['scale_error']) for r in subset):.9f}; mean raw={np.mean([r['rule']['raw'] for r in subset]):.9f}%, max raw={max(r['rule']['raw'] for r in subset):.9f}%.",
            f"  {group}: {len(ordinary)}/{len(subset)} translation-law controls; among these max scale error={max(abs(r['rule']['scale_error']) for r in ordinary):.9f}, max raw={max(r['rule']['raw'] for r in ordinary):.9f}%."]
    lines += ['Exceptions/residuals above 2e-5 scale error (ALL retained in the preceding totals):']
    for r in native:
        if abs(r['rule']['scale_error'])>2e-5:
            lines.append(f"  {r['name']} {r['mode']}: error={r['rule']['scale_error']:+.9f}, raw={r['rule']['raw']:.7f}%, framing matrix error={r['reconstruction_error']:.9g}")
    lines += ['The small ordinary residuals (up to ~4e-4 scale) mean this is an approximation, not an exact clone.',
        'V40S3 Full/Vertical violate Round 7 framing too; scaling cannot repair that independent failure.',
        '', '5. FRAMING INTERACTION / MODE STABILITY',
        'Round 7 centre is affine-equivariant: c(s*q+t)=s*c(q)+t, since its weights sum to one.',
        'Thus F(S_.5(H)) = S_.5(F(H)) for fixed positive s. Size features are translation invariant.',
        'scale_output preserves the input blend centre, so F(scale_output(H))=scale_output(F(H)).',
        'The probe checks this to 2e-14 on all 336 native base maps; tests also cover reversed pitch/yaw/roll.',
        'Framing then scale about (.5,.5) is correct. Scaling about origin AFTER framing displaces the centre',
        'by ((s-1)/2,(s-1)/2); that produces a separate translation error. It is not the implementation.',
        'Scale is NOT a stable constant per mode (population std across native scenes/slots):']
    for group in ('four_scene', 'sweep'):
        for mode in ('auto', 'full', 'level', 'vertical'):
            ss = [r['scale'] for r in native if r['group']==group and r['mode']==mode]
            lines.append(f"  {group:<10} {mode:<8} n={len(ss):3}: min={min(ss):.7f} max={max(ss):.7f} mean={np.mean(ss):.7f} std={np.std(ss):.7f}")
    lines += ['', '6. FRESH END-TO-END CLONE DISTANCE / RESIDUAL LOWER BOUNDS',
        'All 160 fresh defaults are bit-identical to Round 7 H_unframed; pre-scale framed H is bit-identical',
        'to Round 7 H. All 160 fresh opt-in attempts accepted. Rotation, damping, source focal and quality',
        'remain identical. Every shape distance is invariant to <=1e-12. Final raw uses ONLY the runtime rule.',
        'centre-scale LB = OFFLINE best single scale about (.5,.5), with translation framing held fixed.',
        'shape LB = OFFLINE best scale AND translation, no rotation/shear/perspective removal.',
        'No uniform scale can beat centre-scale LB with framing fixed; even free translation cannot beat shape LB.',
        'Oracle scales in this section are diagnostics, never used by the runtime.',
        '', 'prior               scene       mode        raw original  framed   both     applied s  oracle s  centre LB shape LB']
    for prior in ('scene_focal', 'adobe_stored_focal'):
        for scene in ('A_clean', 'B_strong', 'C_noisy', 'D_roll'):
            for mode in ('auto', 'full', 'level', 'vertical'):
                rr = [r for r in measured if (r['prior'], r['scene'], r['mode'])==(prior, scene, mode)]
                mean = lambda key: float(np.mean([r[key] for r in rr]))
                values = [np.mean([r[k]['raw_rms_pct'] for r in rr]) for k in ('before', 'framed', 'final')]
                lines.append(f"{prior:<19} {scene:<11} {mode:<8} {values[0]:12.6f} {values[1]:8.6f} {values[2]:8.6f} {mean('applied_scale'):10.6f} {mean('oracle_centre_scale'):9.6f} {mean('centre_scale_lower_bound'):9.6f} {mean('shape_lower_bound'):8.6f}")
        for mode in ('all', 'auto', 'full', 'level', 'vertical'):
            rr = [r for r in measured if r['prior']==prior and (mode=='all' or r['mode']==mode)]
            raw = [np.mean([r[k]['raw_rms_pct'] for r in rr]) for k in ('before', 'framed', 'final')]
            pooled = [np.sqrt(np.mean([r[k]['raw_rms_pct']**2 for r in rr])) for k in ('before', 'framed', 'final')]
            lb = np.mean([r['centre_scale_lower_bound'] for r in rr])
            shape = np.mean([r['shape_lower_bound'] for r in rr])
            lines.append(f"  {prior} {mode}: mean raw {raw[0]:.6f} -> {raw[1]:.6f} -> {raw[2]:.6f}%; centre LB={lb:.6f}%, shape LB={shape:.6f}%.")
            lines.append(f"    pooled RMS {pooled[0]:.6f} -> {pooled[1]:.6f} -> {pooled[2]:.6f}%; max final={max(r['final']['raw_rms_pct'] for r in rr):.6f}%; original squared-error reduction={100*(1-(pooled[2]/pooled[0])**2):.5f}%.")
            if mode == 'all':
                pooled_lb = np.sqrt(np.mean([r['centre_scale_lower_bound']**2 for r in rr]))
                pooled_shape = np.sqrt(np.mean([r['shape_lower_bound']**2 for r in rr]))
                share = sum(r['final']['raw_rms_pct']**2 for r in rr if r['scene']=='B_strong' and r['mode'] in ('auto','level'))/sum(r['final']['raw_rms_pct']**2 for r in rr)
                lines.append(f"    pooled lower bounds: fixed framing scale={pooled_lb:.6f}%, free translation+scale={pooled_shape:.6f}%. An oracle scale could still remove {100*(1-(pooled_lb/pooled[2])**2):.3f}% of remaining squared displacement; B_strong Auto/Level account for {100*share:.3f}% of current squared displacement.")
        rr = [r for r in measured if r['prior']==prior]
        lines.append(f"  {prior}: wrong origin-pivot AFTER framing mean raw={np.mean([r['wrong_pivot_raw'] for r in rr]):.6f}%.")
    lines += ['Residual: B_strong Auto and Level dominate the remaining distance. Their irreducible shape',
        'errors are already large; changing scale cannot fix their angle/geometry differences. There is also',
        'a remaining scale/framing choice error above the oracle bounds. When returned quad geometry differs,',
        'this geometric scale rule also returns a different zoom from the native one. Lower bounds and raw',
        'distances are not additive components. This work changes no damping.',
        '', '7. IMPLEMENTATION / VALIDATION / LIMITS',
        'port/upright_port.py: scale_output(H); solve/solve_ex(..., framing=True, scaling=True).',
        'Default scaling=False. scale_output also works without framing, preserving the current bounds centre.',
        'H_unscaled retains its input; H_unframed retains the original solve when framing is enabled.',
        'Off/rejected solves stay identity. Finite-input/horizon checks reuse frame_output; degenerate or',
        'non-enclosing quads are rejected. Runtime requires no new dependency and reads no evidence/native data.',
        f"AST check: all {result['unchanged_functions']} prior functions except solve_ex/solve unchanged; module constants/imports unchanged.",
        f"Production SHA256: {result['production_sha256']}",
        'Regression tests were run red before implementation (12 failed: missing scale_output/scaling option).',
        'Reproduce: python3.12 evidence/round8_probe.py > evidence/round8_probe_run.txt 2>&1',
        '           python3.12 -m pytest -q > evidence/round8_pytest.txt 2>&1',
        'round8.json includes every native row, correlations, policy errors and all 160 fresh measurements.',
        'No new real-photo/native execution or broad yaw/roll sweep. Related synthetic data only; 1.10 was',
        'identified using these stored maps. Generality beyond them is unverified, hence a separate opt-in.',
        'All edits and generated artifacts are inside research/upright/.']
    path = ROOT/'evidence/round8_pytest.txt'
    lines += path.read_text().splitlines()[-4:] if path.exists() else ['Full post-change pytest pending.']
    (ROOT/'evidence/round8.txt').write_text('\n'.join(lines)+'\n')


def main():
    rows = dataset()
    recovered_rule(rows)
    result = dict(rows=rows, correlations={}, hypotheses={})
    for group in ('four_scene', 'sweep'):
        subset = [r for r in rows if r['group']==group]
        result['correlations'][group] = {m: correlations([r for r in subset if m=='all' or r['mode']==m])
                                         for m in ('all', 'auto', 'full', 'level', 'vertical')}
        result['hypotheses'][group] = hypotheses(subset)
    result['measurements'] = measure()
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    original = ast.parse((ROOT/'evidence/round8_original_port.py').read_text())
    current = ast.parse((ROOT/'port/upright_port.py').read_text())
    old, new = functions(original), functions(current)
    unchanged = [k for k in old if k not in ('solve_ex', 'solve')]
    assert all(old[k]==new[k] for k in unchanged)
    globals_dump = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body if not isinstance(n, ast.FunctionDef)]
    assert globals_dump(original)==globals_dump(current)
    result.update(unchanged_functions=len(unchanged), production_sha256=hashlib.sha256((ROOT/'port/upright_port.py').read_bytes()).hexdigest())
    (ROOT/'evidence/round8.json').write_text(json.dumps(result, indent=2)+'\n')
    report(result)
    for group, values in result['hypotheses'].items():
        print(group)
        for k, v in values.items():
            print(k, v)
    print('native scale and boundary metrics')
    for r in rows[:16]:
        print(r['name'], r['mode'], 's', round(r['scale'], 8), 'fit', {k: round(v, 7) for k, v in policies(r).items()})


if __name__ == '__main__':
    main()
