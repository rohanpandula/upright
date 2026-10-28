"""Rebuild Round 10 evidence from fresh per-attempt outputs and frozen policy."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'evidence'),str(ROOT/'port'),str(ROOT)]
from round10_analyze import load,attempts,decision,distribution,summary,calibration


def scores(rows,key='both'):
    return [r['identity']['raw_rms_pct'] if not r['candidate_ok'] else
            r['candidate' if key=='both' else 'bare']['raw_rms_pct'] for r in rows]


def stats_line(label,rows):
    d=distribution(scores(rows))
    return f"{label:29} {len(rows):5d} {sum(r['candidate_ok'] for r in rows):5d} " + ' '.join(
        f"{d[k]:11.6f}" for k in ('median','p90','p99','max')) + ' '+ ' '.join(f"{d[k]:5d}" for k in ('over1','over5','over20'))


def main():
    calibration()
    candidates=load('round10_candidates.jsonl');actual=load('round10_actual.jsonl')
    assert len(candidates)==len(actual)==679
    target={p['image_id']:p for p in candidates}
    assert len(target)==679
    for p in actual:
        b=target[p['image_id']]
        if 'error' in p:
            assert p['error']==b['error'];continue
        for a,c in zip(p['rows'],b['rows']):
            passed=decision(c,1.,'relative_1.0')
            assert bool(a['quality'] and a['quality']['passed'])==passed
            assert a['candidate_ok']==(passed and c['candidate_ok'])
            if a['candidate_ok']:
                np.testing.assert_array_equal(a['H'],c['H'])
                assert a['candidate']==c['candidate']
            else:np.testing.assert_array_equal(a['H'],np.eye(3))
    cr=attempts(candidates);ar=attempts(actual)
    assert len(ar)==1356
    assert not any(r['quality'] and r['quality']['scatter_passed'] and r['quality']['bias_passed'] for r in cr)
    groups={
        'Adobe-corrected 450 photos':[r for r in ar if r['cohort']!='abstain'],
        'Original 300 photos':[r for r in ar if r['cohort']=='round9'],
        'New 150 photos (untouched)':[r for r in ar if r['cohort']=='new_holdout'],
        'Validation corrected photos':[r for r in ar if r['split']=='validation' and r['cohort']!='abstain'],
        'Adobe-abstained 228 photos':[r for r in ar if r['cohort']=='abstain'],
        'All decoded 678 photos':ar,
        'Accepted only (diagnostic)':[r for r in ar if r['candidate_ok']],
    }
    result=dict(policy=json.loads((ROOT/'evidence/round10_policy.json').read_text()),
                errors=[p for p in actual if 'error' in p],
                production_sha256=hashlib.sha256((ROOT/'port/upright_port.py').read_bytes()).hexdigest(),
                groups={},confusion={})
    for name,rows in groups.items():
        result['groups'][name]=dict(both=distribution(scores(rows)),bare=distribution(scores(rows,'bare')),
            identity=distribution([r['identity']['raw_rms_pct'] for r in rows]),accepted=sum(r['candidate_ok'] for r in rows),
            by_mode={m:dict(error=distribution(scores([r for r in rows if r['mode']==m])),accepted=sum(r['candidate_ok'] for r in rows if r['mode']==m)) for m in ('auto','full')})
    for label,subset in [('calibration',[r for r in cr if r['split']=='calibration']),
                         ('validation',[r for r in cr if r['split']=='validation']),('all',cr),
                         ('corrected',[r for r in cr if r['cohort']!='abstain'])]:
        result['confusion'][label]=summary(subset,1.,'relative_1.0')
    for label,rows in groups.items():
        ids={r['image_id'] for r in rows}
        result['groups'][label]['photos_any_over']={str(t):sum(any(v>t for v in scores([r for r in rows if r['image_id']==iid])) for iid in ids) for t in (1,5,20)}
    q=load('round10_candidates.jsonl')
    quality=[r['quality'] for p in q if p['cohort']=='round9' for r in p.get('rows',[]) if r['quality']]
    result['original_scatter_quantiles']=np.percentile([max(r['family_median_deg']) for r in quality],[0,50,90,99,100]).tolist()
    rows=json.loads((ROOT/'evidence/round10_sweep.json').read_text());assert len(rows)==640
    assert all(r['ok'] and r['matrix_maxdiff']==0 for r in rows)
    result['sweep']={name:dict(attempts=40,max_axis=np.max([r['axis'] for r in rows if r['scene']==name and r['mode']=='full'],axis=0).tolist()) for name in dict.fromkeys(r['scene'] for r in rows)}
    quality=json.loads((ROOT/'evidence/round10_quality.json').read_text())
    result['quality']={}
    for group in ('fixtures','heldout'):
        trials=[t for s in quality['scenes'].values() if s['group']==group for t in s['trials']]
        assert all(t['ok'] and t['gate_pass'] and max(t['candidate_axis_deg'])<.1 for t in trials)
        a=np.array([t['returned_axis_deg'] for t in trials])
        result['quality'][group]=dict(n=len(trials),mean_family=float(a.mean()),worst=float(a.max()),p99=float(np.percentile(a.max(axis=1),99)))
    old=json.loads((ROOT/'evidence/round6_adobe.json').read_text())
    new=json.loads((ROOT/'evidence/round10_adobe_bare.json').read_text());assert old==new
    assert len(json.loads((ROOT/'evidence/round10_heldout.json').read_text()))==280
    geo=json.loads((ROOT/'evidence/round10_geometry.json').read_text())
    geo_old=json.loads((ROOT/'evidence/round9_sweep.json').read_text())
    assert all(geo[k]==v for k,v in geo_old.items())
    result['geometry']=dict(band_cases=len(geo['rows']),known_family_errors=len(geo['family_failures']),stress_accepted=len(geo['stress_accepted']),stress_rejected=len(geo['stress_rejections']),exactly_equals_round9=True)
    result['reasons']=dict(Counter(r['reason'] for r in ar))
    result['floor_comparison']={}
    cal=[r for r in cr if r['split']=='calibration']
    for bias in ('old','relative_1.0','none'):
        selected=[r for r in cal if r['quality'] and decision(r,max(.2,r['quality']['limit_deg']),bias)]
        result['floor_comparison'][bias]=dict(gate_accept=len(selected),good_le1=sum(r['candidate'] is not None and r['candidate']['raw_rms_pct'] is not None and r['candidate']['raw_rms_pct']<=1 for r in selected))
    assert result['floor_comparison']['old']['gate_accept']==0
    assert result['floor_comparison']['relative_1.0']==dict(gate_accept=2,good_le1=0)
    result['before_sha256']=hashlib.sha256((ROOT/'evidence/round10_before_port.py').read_bytes()).hexdigest()
    assert result['before_sha256']=='60f842ea1d03356f4c411b00f3e0e0802dc0a0a8f20a86a468268979fa7b14ad'
    result['invalid_fitted_candidates']=sum(r['quality'] is not None and (r['candidate'] is None or r['candidate']['raw_rms_pct'] is None) for r in cr)
    assert result['invalid_fitted_candidates']==174
    result['returned_confusion']={}
    for t in (1,5,20):
        counts={k:0 for k in ('accept_good','accept_bad','reject_good','reject_bad')}
        for r,e in zip(ar,scores(ar)):
            counts[('accept_' if r['candidate_ok'] else 'reject_')+('bad' if e>t else 'good')]+=1
        result['returned_confusion'][str(t)]=counts
    (ROOT/'evidence/round10_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    d=result['groups']['Adobe-corrected 450 photos']['both']
    lines=[
        'ROUND 10 — REAL-PHOTO GATE CALIBRATION — 2026-10-28', '='*116,
        'HEADLINE: STILL A POOR REAL-PHOTO CLONE. The resolution-dependent gate is fixed, but coverage remains very low.',
        f"450 Adobe-corrected real photos / 900 Auto+Full attempts, framing+scale ON: median {d['median']:.6f}%, p90 {d['p90']:.6f}%,",
        f"p99 {d['p99']:.6f}%, max {d['max']:.6f}%; >1/5/20% = {d['over1']}/{d['over5']}/{d['over20']}. Accepts 80/900 (8.89%); 820 return identity.",
        'Original 300-photo cohort: accepts 58/600, median 2.126793%, p90 7.979433%, p99 16.315469%, max 24.099793%.',
        'New untouched 150-photo cohort: accepts 22/300, median 1.987409%, p90 7.850479%, p99 17.448677%, max 81.961591%.',
        'Two accepted real attempts exceed 5%. Restoring non-identity output does not establish usable Adobe equivalence.',
        '', '1. METRIC, SPLITS, AND ACTUAL EXECUTION',
        'Metric: RMS displacement on the fixed 25x25 source grid, including corners, divided by original source diagonal,',
        'times 100. No fitted translation/scale/crop/rotation is removed. Percentiles use NumPy linear interpolation.',
        'Native maps and returned maps are compared in sensor axes; pixel weights use unrotated catalog width/height.',
        'Orientation Q is applied once to detections and removed by Q^-1 H Q for scoring. TIFF storage pixels use tifffile.',
        'RAW uses the established half-size camera-WB decoder/crop; detection uses a 1200px working size. Supports are',
        'rescaled to original catalog dimensions before solving. No Adobe preview, lens model, or native GUI solve is used.',
        'The older real_photos_raw/abstain scores are not reused for calibration; every source was decoded/detected again.',
        '',
        'Frozen sample: original round9_sample.json (300) + existing identity-target cohort (229) + 150 new paths drawn',
        'with seed 10010 from the available catalog population, excluding the first two cohorts. Catalog access was read-only.',
        'Calibration was first 150 original + first 114 abstention photos (264 selected). Validation was the remaining',
        '150 original +115 abstention photos +150 newly drawn photos (415). Thresholds were frozen before validation scores.',
        'One calibration abstention DNG fails with LibRawTooBigError: image_id=2116098, DSCF2505.DNG. It is retained as',
        'an explicit decode-error record, not assigned an identity or a numeric score, and not replaced. Thus calibration',
        'has 263 decoded photos/526 attempts; validation has 415/830; total decoded 678/1356 from 679 selected photos.',
        'The headline uses the 450 photos Adobe corrected. The 228 identity-target photos are reported separately, so',
        'their 456 zero-error identity returns do not conceal poor correction coverage in the headline.',
        '',
        'Candidate run bypasses only the gate decision, keeping diagnostics, to score rejected candidates with framing',
        'and scale. Invalid candidate maps are recorded, not silently scored as identity. A second fresh solve_ex run',
        'uses the shipped gate, framing=True, scaling=True, seed=0, both modes. Its 679 records match the frozen policy',
        'simulation exactly; all 80 accepted H and scores are identical to their ungated candidates. All 1276 rejected',
        'attempts return exact identity and remain in returned-error statistics. Detection supports are cached from this',
        'round for that second solve run. Oracle transforms enter scoring only, never fitting or the production gate.',
        '', '2. WHAT THE GATE MEASURES AND WHAT CHANGED',
        'The old scatter limit was asin(min(1,12/(maxdim-1))). It was 0.491464 deg at 1400px and 0.079467 deg at 8653px.',
        'That normalized optimizer noise scale is not an image-resolution-independent test of structural consistency.',
        'Round 3 also required abs(signed median)+margin <=0.1 deg. Every fitted original real-photo attempt fails that',
        'bias condition, even with the scatter gate removed. An absolute .2-degree scatter floor alone cannot fix this.',
        '',
        'For EACH selected undamped family, retain source-length-weighted pixel-angle statistics:',
        '  a = weighted median absolute angle, degrees',
        '  b = absolute weighted median signed angle, degrees',
        '  n_eff = sum(length)^2 / sum(length^2)',
        '  u = 3.72 * weighted MAD(signed angle) / sqrt(n_eff)',
        'New acceptance requires both families to satisfy:',
        '  a <= 1.0 deg AND (b + u <= 0.1 deg OR b + u < a).',
        'The second inequality is strict: coherent .11/.13-degree bias with zero MAD must still fail. A noisy, centered',
        'family can pass when its bias plus support margin is smaller than its total observed scatter. Precise families',
        'retain the .1-degree absolute budget. The 1-degree cap prevents very broad/poorly classified families from',
        'passing merely because their signed errors cancel. These are empirical consistency checks, not camera-error',
        'bounds or calibrated probabilities. Effective count does not account for correlated or duplicated chains.',
        '',
        'Both tests are invariant to uniform pixel resizing for fixed H and normalized supports. Pixel dimensions no',
        'longer enter acceptance; source-length weights and n_eff are unchanged by uniform scaling. worst_scale stays',
        'as a legacy diagnostic, but is never used to accept/reject. limit_deg is now 1.0; family_bias_limit_deg exposes',
        'the per-family max(.1,a) budget; bias_limit_deg=.1 remains the legacy floor diagnostic.',
        'Only _alignment_quality changed in production (44 other functions and module globals/imports unchanged by AST).',
        'No detector, VP selector, focal policy, optimizer, loss, damping, framing, scale switch, or mode constant changed.',
        '', '3. CALIBRATION EVIDENCE AND REJECTED ALTERNATIVES',
        '264 selected /263 decoded calibration photos: 526 attempts, 354 fitted candidates; no-fit cases are retained.',
        'Table uses the chosen two-part bias rule and changes only the absolute scatter cap. Good means candidate raw',
        '<=1%; bad means >5% (invalid maps also count bad). Accepted entries below are all finite valid maps.',
        'scatter cap deg    accepted    accepted <=1%    accepted >5%    returned max %',
    ]
    table=json.loads((ROOT/'evidence/round10_calibration.json').read_text())
    for r in table:
        if r['bias']=='relative_1.0' and r['scatter'] in (.2,.5,.75,1.,1.5,2.,3.,5.,10.):
            lines.append(f"{r['scatter']:15.2f} {r['gate_accept']:11d} {r['confusion']['1']['accept_good']:16d} {r['confusion']['5']['accept_bad']:15d} {r['error']['max']:17.6f}")
    lines += [
        'Choice: 1 degree is the largest tested cap with zero >5% accepted calibration candidates. The .2-degree floor',
        'max(old limit,.2 deg) admits only two attempts with the new bias rule, neither <=1%; retaining the old',
        'bias rule admits zero. This floor does not restore useful coverage. A 1.5-degree cap introduces a',
        '41.099070% error. This is a conservative discrete choice, not a recovered Adobe threshold or an optimum.',
        'The additional 150-photo cohort and remaining original/abstention photos did not set or revise this constant.',
        '',
        'Alternatives (full tables in round10_calibration.json): retaining the old upper-bias limit accepts zero at',
        'EVERY scatter cap. Subtracting the margin from bias, or merely checking statistically resolved bias, restores',
        'coverage but fails existing recorded H09-tail regressions. A pure scatter rule accepts coherent synthetic bias.',
        'Bias budgets at .25/.5/.75 times absolute scatter reduce coverage; the full-scatter budget preserves precise',
        'bias rejection with strict inequality and adds real good candidates. Budgets above scatter would admit coherent',
        'bias even with zero MAD. Direct a*sqrt(n_eff) limits were also tabulated: support count alone does not distinguish',
        'a wrong family, and fragmented/correlated supports are not independent measurements. Count therefore stays in',
        'the uncertainty margin rather than setting the absolute scatter cap.',
        'A fixed original-pixel residual would keep the resolution defect. A pixel residual at a declared detector',
        'reference size could be invariant, but would be another length-dependent angular policy; it was not measured',
        'as an alternative here. The selected statistic directly measures orientation consistency in isotropic pixels.',
        '', '4. CONFUSION MATRICES — FROZEN GATE VS ADOBE CLONE ERROR',
        'Rows below classify the ATTEMPTED framing+scale candidate, including rejected candidates. The 174 geometrically',
        'invalid fitted candidates get their own column. The 464 attempts with no fit have no candidate error; they are',
        'shown separately, and their returned identities are still included in all headline distributions.',
        'All decoded photos, 892 fitted attempts:',
        'gate decision       candidate <=5%       candidate >5%       invalid candidate',
        'accept                          78                   2                       0',
        'reject                         317                 321                     174',
        'No fitted candidate: 464. Finite candidate matrix uses 718 attempts; full fitted matrix uses 892.',
        'At <=1%: accept good/bad =47/33; reject good/bad=112/526, plus 174 invalid and 464 no-fit.',
        'At <=20%: accept good/bad=80/0; reject good/bad=537/101, plus 174 invalid and 464 no-fit.',
        'Thus finite <=5% candidate recall is only 78/395=19.75%. Accepted <=5% precision is 78/80=97.5%; these descriptive',
        'rates are correlated by mode and capture, not independent-probability estimates.',
        '',
        'Calibration, 354 fitted: accept <=5/>5/invalid=30/0/0; reject=117/130-minus-invalid/invalid (see JSON).',
        'Untouched validation, 538 fitted: accept <=5/>5/invalid=48/2/0. Full split matrices, with invalid counted bad,',
        'and thresholds 1/5/20 are in round10_summary.json. No threshold was revised after the two validation false accepts.',
        '',
        'Separate confusion against the FINAL RETURNED error, all 1356 scored attempts (identity fallbacks included):',
        'gate/output decision       returned <=5%       returned >5%',
        'accept                                78                  2',
        'reject                              1099                177',
        'This second matrix measures the actual returned clone, not quality of rejected candidates. Rejection does not',
        'make a distorted source upright, and numerous small identity errors reflect tiny Adobe corrections.',
        '', '5. GATE VS WHETHER ADOBE ABSTAINED',
        'Adobe abstention here means the stored slot is identity to max-coefficient tolerance 1e-6; there is no native',
        'confidence/reason field. Original/new corrected cohorts require nonidentity Auto AND Full; the separate 229',
        'abstention cohort has identity targets. One abstention photo failed decoding.',
        'All 1356 scored attempts:     Adobe corrected       Adobe abstained',
        'gate accepts                               80                     0',
        'pipeline rejects                          820                   456',
        'Among fitted candidates only:             corrected             abstained',
        'gate accepts                               80                     0',
        'gate rejects                              564                   248',
        'Remaining no-fit attempts: 256 corrected,208 abstained. Agreement with identity targets is good, but correction',
        'recall is only 8.89% of attempts/40 of 450 photos. This gate is still far too restrictive or uninformative for',
        'most ordinary photographs Adobe corrects; it does not fulfill broad useful-correction coverage.',
        '', '6. REAL-PHOTO RETURNED DISTRIBUTIONS — FRAMING + SCALE ON',
        'Counts are ATTEMPTS (each photo supplies Auto and Full). Every rejected identity is included.',
        'cohort                            n    ok      median%        p90%        p99%        max%   >1%   >5%  >20%',
    ]
    for name,rows in groups.items():lines.append(stats_line(name,rows))
    lines += ['', '450 corrected photos by mode:',stats_line('Auto',[r for r in groups['Adobe-corrected 450 photos'] if r['mode']=='auto']),stats_line('Full',[r for r in groups['Adobe-corrected 450 photos'] if r['mode']=='full']),
        '', 'Counts of PHOTOS with either mode over each threshold (>1/>5/>20):']
    for name in ('Original 300 photos','New 150 photos (untouched)','Adobe-corrected 450 photos'):
        d=result['groups'][name]['photos_any_over'];lines.append(f"  {name}: {d['1']}/{d['5']}/{d['20']}")
    lines += ['', 'Fresh original-300 accepted-gate comparison (same decisions; no fitted alignment removed):']
    for label,key in [('without framing/scale','bare'),('with framing+scale','both'),('identity baseline','identity')]:
        d=result['groups']['Original 300 photos'][key];lines.append(f"  {label:24} median={d['median']:.6f}, p90={d['p90']:.6f}, p99={d['p99']:.6f}, max={d['max']:.6f}; >1/5/20={d['over1']}/{d['over5']}/{d['over20']}")
    lines += [
        'Accepted-only statistics are diagnostic and must not replace the complete-distribution headline.',
        '', '7. REAL TAILS AND THE FOUR CITED FILES',
        'False accepts: id4583002 Auto=5.701816% vs identity=3.371626%; id4583003 Auto=5.732568% vs identity=3.371626%.',
        'They are related R0000209 DNG copies/imports (different paths/catalog IDs), so these are two attempts, not two',
        'independent captures. Both have small selected-family residuals but wrong Adobe agreement. Their Full attempts',
        'remain <=5%. Internally consistent families do not identify the correct scene/camera or Adobe response.',
        'Worst returned result: id5722313 /DSCF7746.RAF Full=81.961591%, rejected to identity. Its ungated candidate is',
        'also bad (80.662360%); family medians=.946600/2.981173 deg. The gate cannot recover a missing correct solution.',
        '',
        'The four cited TIFFs were additionally loaded from the read-only catalog. At original dimensions:',
        '  DSCF3578: medians 2.441021/2.007433 deg -> rejects both old/new; ungated Auto/Full=2.561367/5.326036%.',
        '  DSCF3582: 30.139034/1.436023 deg -> rejects both; ungated Auto/Full=1.192118/4.591007%.',
        '  DSCF3586: insufficient line structure at this working detector size -> rejects before quality.',
        '  DSCF3590: .898281/4.850186 deg -> rejects both.',
        'These do not become acceptable merely by removing the .079-degree gate. With identical cached normalized',
        'supports expressed at quarter source dimensions, DSCF3590 becomes accepted (.874476/.092813 deg); the other',
        'three remain rejected. This test changes solver pixel dimensions, not just the gate; RANSAC/calibration still',
        'contain pixel thresholds. It is distinct from the default detector resize that rescales supports back to the',
        'original dimensions. Full old/new quarter/original decisions are in round10_repro_decisions.json.',
        'Two other real support sets (R0006607.DNG and L1005986.DNG) now accept at original AND quarter dimensions in',
        'both modes. Their real cached supports form the regression fixture; no original photo volume is needed by tests.',
        '', '8. SYNTHETIC NONREGRESSION, ADOBE, AND TESTS',
        '40 seeds x4 original fixtures x4 automatic modes =640 returned framing+scale mappings; every H, H_unscaled and',
        'H_unframed is bit-for-bit equal to the saved pre-change implementation. Detector arrays are exactly equal.',
        'Full-mode independent V/H maxima over seeds 0..39:']
    for scene,d in result['sweep'].items():lines.append(f"  {scene:12} {d['max_axis'][0]:.9f}/{d['max_axis'][1]:.9f} deg")
    lines += [
        'Quality/tail sweep: 400 fixture/focal trials +700 held-out Full trials (14 scenes x50 seeds); all pass with',
        'independent family error <.1 degree. No failed/no-fit case is dropped. Detailed tails:']
    for name,d in result['quality'].items():lines.append(f"  {name:10} n={d['n']}, mean family={d['mean_family']:.9f} deg; p99 max-family={d['p99']:.9f}; worst={d['worst']:.9f}")
    lines += [
        'All 280 held-out all-mode/seed framing+scale H are exactly equal to round9_heldout.json.',
        'Original BAD seed-4 theta with its ORIGINAL pre-round1 family selector/supports is reconstructed separately:',
        '  family medians=2.707385072/.496885301 deg; worst_scale=16.520511568; scatter=False,bias=False.',
        '  Public API returns ok=False and exact identity in Auto/Full/Level/Vertical. Production uses no replay/truth.',
        'The existing repaired-family bad-theta replay, H09-tail replays, precise .11/.13-degree bias controls, impossible',
        'camera warp and undamped-before-mode checks all remain in the unchanged tests and pass.',
        '',
        'Adobe comparison: fresh 320 bare benchmark records exactly equal Round 6, including both priors/all modes.',
        'Fresh framing+scale comparison: all 160 mappings and scores exactly equal Round 8:',
        '  scene-focal mean=.587091572%, median=.316547644%, max=2.605488274% (80 attempts);',
        '  stored-focal mean=.584356996%, median=.319487189%, max=2.606536762% (80 attempts).',
        'These synthetic means are controls, not the headline.',
        '',
        'Accuracy benchmark: every current-detector score is unchanged. Frozen old-detector B_strong now accepts:',
        '  returned error improves from 11.860099/5.417679 deg (identity) to .039997/.028217 deg.',
        'Sensitivity benchmark: all 180 attempted candidate errors are unchanged. Fourteen previously rejected outputs',
        'now accept, and every changed returned V/H component improves versus its previous identity result; no returned',
        'axis metric regresses. Seven of those exploratory outputs still exceed .1 degree in independent truth, including',
        'least-squares D_roll at .031053/2.303455 deg and least-squares 01_input at .065409/.551026 deg. The gate is not',
        'a universal truth-error detector. These alternative losses are not the production default.',
        '',
        'Round 9 scale/quad sweep re-run: all old JSON fields exactly equal; 57,036 band cases,20 known outside-band',
        'family errors; random stress accepts3281/rejects6719. The exact-diamond ZeroDivisionError and 1.10 branch',
        'discontinuity remain unchanged. No unrelated geometry repair is bundled into this gate calibration.',
        'Before-change suite:99 passed (49.58s). Two added regression tests fail on the old gate (round10_red.txt).',
        'Final suite:101 passed (51.60s), existing pytest-asyncio configuration deprecation warning only.',
        '', '9. ARTIFACTS, REPRODUCTION, AND LIMITS',
        f"Production SHA256: {result['production_sha256']}",
        f"Frozen sample SHA256: {hashlib.sha256((ROOT/'evidence/round10_sample.json').read_bytes()).hexdigest()}",
        'Production change:port/upright_port.py::_alignment_quality. Docs:port/README.md. Tests:port/test_round10.py',
        'and evidence/round10_regression_supports.npz. All existing tests remain unchanged.',
        'round10_sample.json and round10_policy.json freeze cohorts/choice. round10_candidates.jsonl records ungated',
        'candidates/invalid maps; round10_actual.jsonl records actual public returned H/reasons/metadata/scores.',
        'round10_summary.json contains full mode/cohort/split distributions, both confusion definitions and image counts.',
        'round10_calibration.json, round10_sweep.json, round10_quality.json, round10_adobe*.json, round10_heldout.json,',
        'round10_geometry.json and round10_sensitivity*.json retain detailed checks. round10_supports/ caches this round\'s',
        'detected supports. Historical evidence files are preserved.',
        '',
        'From research/upright/ (Python3.12, installed dependencies; set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):',
        '  python3.12 evidence/round10_probe.py collect --workers 3',
        '  python3.12 evidence/round10_analyze.py',
        '  python3.12 evidence/round10_probe.py actual --workers 3',
        '  python3.12 evidence/round10_validate.py sweep',
        '  python3.12 evidence/round10_validate.py heldout',
        '  python3.12 evidence/round10_validate.py adobe',
        '  python3.12 evidence/round10_validate.py bad',
        '  python3.12 evidence/round10_validate.py scope',
        '  python3.12 evidence/round10_validate.py geometry',
        '  python3.12 port/benchmark_quality.py --seeds 50 --output evidence/round10_quality.json',
        '  python3.12 port/benchmark_adobe.py --seeds 5 --output evidence/round10_adobe_bare.json',
        '  python3.12 port/benchmark_accuracy.py --output evidence/round10_accuracy.json',
        '  python3.12 port/benchmark_sensitivity.py --record-failures --output evidence/round10_sensitivity.json',
        '  python3.12 -m pytest -q',
        '  python3.12 evidence/write_round10_report.py',
        'For the initial old-gate candidate diagnostics, round10_before_port.py is the immutable pre-change snapshot.',
        'Do not re-freeze sample with the sample stage if reproducing these exact cohorts.',
        '',
        'Limits: availability-biased personal catalog; split is by catalog path, not scene/capture/content hash; copies',
        'and related scenes can cross splits. Only one seed per real photograph; no independent real-world line truth;',
        'RAW/lens/preview pipeline is not matched to Adobe. One decode failure is unscored. Identity targets are a proxy',
        'for native abstention. Weighted uncertainty assumes more independence than fragmented chains may provide.',
        'No native Adobe invocation, catalog write, or external publish occurred. All edits/artifacts are within research/upright/.',
        'The resolution-dependent rejection is repaired and selected ordinary photographs now correct. Broad acceptance',
        'and reliable Adobe cloning remain unsolved. The real-photo distribution above is the current accuracy result.',
    ]
    # Replace the split shorthand with explicit finite/invalid counts.
    idx=lines.index('Calibration, 354 fitted: accept <=5/>5/invalid=30/0/0; reject=117/130-minus-invalid/invalid (see JSON).')
    for label in ('calibration','validation'):
        subset=[r for r in cr if r['split']==label]
        invalid=sum(r['quality'] is not None and (r['candidate'] is None or r['candidate']['raw_rms_pct'] is None) for r in subset)
        m=result['confusion'][label]['confusion']['5']
        line=f"{label.capitalize()}, {sum(r['quality'] is not None for r in subset)} fitted: accept <=5/>5/invalid={m['accept_good']}/{m['accept_bad']}/0; reject={m['reject_good']}/{m['reject_bad']-invalid}/{invalid}."
        if label=='calibration':lines[idx]=line
        else:lines[idx+1]=line+' Full split matrices, with invalid counted bad,'
    (ROOT/'evidence/round10.txt').write_text('\n'.join(lines)+'\n')
    print('wrote evidence/round10.txt and evidence/round10_summary.json')


if __name__=='__main__':main()
