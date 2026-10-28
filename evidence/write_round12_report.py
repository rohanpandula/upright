"""Write the final Round-12 verdict from frozen numerical evidence."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'


def main():
    s=json.loads((E/'round12_summary.json').read_text());raw=s['groups']['raw387'];gap=s['full_gap']
    lines=[
        'ROUND 12 — VP / MANHATTAN FAMILY SELECTION — 2026-10-28', '='*112,
        'RESULT: DID NOT CLOSE THE ORACLE GAP. No selector change ships.',
        f"Full all-387 RAW candidate median {gap['before']:.6f}% -> {gap['candidate']:.6f}% (oracle {gap['oracle']:.6f}%).",
        f"Oracle reduction was {gap['oracle_reduction_pct']:.4f}%; actual reduction is {gap['reduction_pct']:.4f}%.",
        f"Fraction of that gap closed = (before-after)/(before-oracle) = {gap['gap_closed_pct']:.4f}%: zero positive recovery.",
        'The final candidate improves support purity, coverage, and identity win counts, but Full still loses more',
        'often than it wins on valid maps, its p90 worsens, and a new bad Auto acceptance appears on an abstention',
        'control. The no-regression constraint therefore rules it out. Production was restored byte-for-byte.',
        'The deployable result remains the Round-11 baseline, not the experimental results below.',
        '',
        '1. PROTOCOL / SPLITS / WHAT WAS HELD FIXED',
        'Read Round 11, especially section 5, report section 3.5g, vp_clustering.txt and vp_resolution.txt.',
        'Reuse ONLY frozen Round-10 segment caches, original dimensions, focal metadata, catalog orientation Q,',
        'seed 0, both modes, framing=True, scaling=True. No detection, decoding, rendering, lens, quality-gate,',
        'damping, objective, robust VP refinement, downstream source calibration, framing or scale changes.',
        'Metric: fixed 25x25 source grid RMS / original sensor diagonal, percent; no fitted alignment is removed.',
        'Score Q^-1 H Q in sensor axes. Native H is used only after solving, for scores and the support audit.',
        'Candidate experiment bypasses only the quality verdict; all geometric validity checks still apply.',
        'Returned scores apply the original verdict captured before bypass; every rejection/failure is identity.',
        'W/L/T use 1e-9 percentage-point tolerance. All-attempt wins divide by 387; decisive wins divide by W+L.',
        'Do not equate an increase in maps or gate accepts with an increase in accuracy.',
        '',
        'Full cohort: 450 corrected photographs, including the same 387 RAW files; 228 decoded identity-target',
        'abstention controls are separate. All 679 frozen records were replayed, with 678 cached support sets.',
        'id2116098 had the known Round-10 LibRawTooBigError and no cache; it remains an explicit missing-cache',
        'error record, not a numeric identity result and not a replacement photo. None of the 387 RAW targets fail.',
        'Selector exploration used the 150 corrected calibration photos (127 RAW, 23 raster), not a new sample.',
        'The other 300 corrected photos include 260 RAW. Initial candidate v8 was frozen before full evaluation.',
        'v9 fixes an existing wrong-focal regression exposed by the test suite; its choice did not optimize real',
        'validation scores. The validation set had already been viewed for v8, so v9 is NOT an untouched holdout.',
        'Synthetic held-out scenes are the established 14 fixtures, not newly sampled scenes.',
        '',
        '2. SELECTOR IMPLEMENTATION / PRECISE LIMITS',
        'Retained self-contained implementation: evidence/round12_candidate_port.py (final, focal-aware v9).',
        'Initial fully measured implementation: evidence/round12_candidate_v8_port.py.',
        'Changes are confined to estimate_vanishing_points, _classify, _base_solve, plus _manhattan_margin.',
        'AST scope checks verify all other functions, including detector and quality decision, are unchanged.',
        '',
        'a. Normalize segment line duals in centered isotropic coordinates. Build VP consensus hypotheses from',
        '   dual-plane probes and each dual\'s tangent at infinity. Vote on all existing segments, require >=3',
        '   supports, merge identical vote sets retaining the best representative, and rank by newly explained',
        '   support. Refine VP consensus planes; suppress geometrically duplicate poles; cap accepted VPs at 9.',
        'b. Require >=2 surviving VP families. Assign each segment disjointly to its minimum geometric residual',
        '   VP, with a 1.5-degree angular residual ceiling; repeat assignment/refinement three times. No segment',
        '   is retained or discarded by a vertical/horizontal majority label.',
        'c. Enumerate compatible VP pairs and infer the third Manhattan axis; require finite nonzero frame axes.',
        '   Rank frame hypotheses jointly, rather than independently picking the strongest vertical and the most',
        '   lateral horizontal. An observed depth family is used only when it agrees with the inferred frame.',
        'd. Apply BOTH runner_up/best <= .75 AND best-runner_up > 1.5, with positive finite score checks. The',
        '   runner-up is a distinct frame after 5-degree geometric mode suppression, rather than another nearby',
        '   pole for the same frame. Boundary tests cover ratio equality and strict absolute-margin equality.',
        'e. Determine vertical/lateral identities from the camera rays of the estimated frame. This handles roll',
        '   before applying axis labels; tests recover the correct support families at rolls +/-25 and +/-40 deg',
        '   with perspective and multiple seeds. It retains the closest-upright Manhattan labelling convention;',
        '   a 90-degree axis permutation is not identifiable from line geometry alone without gravity/semantics.',
        'f. v8 incorrectly tested orthogonality at the prior focal before downstream calibration could run. v9',
        '   tests the focal implied by centered perpendicular VPs, when positive and within .5..2 times the prior,',
        '   with a log-focal prior penalty. Existing source calibration still determines the actual solve focal.',
        '',
        'This is closer to the documented stage structure, NOT an exact native VP implementation. In particular,',
        'dual-plane pair probes, angular gates, pole/frame suppression, length weights, and positive product frame',
        'score are CHOICES. Native per-segment dual preparation and signed vote/triple-score units have not been',
        'reconstructed end-to-end. The two margin inequalities are exact; applying them to our score does not',
        'establish native ambiguity calibration. Literal per-segment tangent-seed variants were also tried (v5/v6)',
        'and failed to improve Full. No claim of having reproduced the native per-segment candidate kernel is made.',
        '',
        '3. FULL COLLAPSE — MEASURED MECHANISM, SAME 387 RAW SUPPORT SETS',
    ]
    for name in ['before','candidate']:
        d=s['diagnostic'][name]
        lines.append(f"{name:10} poses={d['selected']}, valid Full maps={d['maps']}, horizon failures={d['horizon']}; "
                     f"Euler-vector norm median/p90={d['rotation_norm_p50_p90_deg'][0]:.3f}/{d['rotation_norm_p50_p90_deg'][1]:.3f} deg")
        lines.append(f"           median length-weighted V/H purity={100*d['purity_median_VH'][0]:.2f}/{100*d['purity_median_VH'][1]:.2f}%; "
                     f"both families >=80% pure: {d['both_purity_ge80']}")
    common=raw['full_common_maps']
    lines += [
        'Purity means selected support length whose native-Full output angle lies within 1 degree of its assigned',
        'axis. It is an Adobe-assisted diagnostic, not physical world-line truth and not a selection input.',
        'The old fixed image-axis majority filter mixes/discards supports under roll and perspective. Independent',
        'family ranking also admits incompatible/spurious VPs. Full applies the entire erroneous pose; Auto',
        'attenuates pitch/yaw. Full therefore crosses the horizon much more often. The new frame selector reduces',
        'that particular failure, but horizontal purity is still below 50% at its median. Surviving wrong axes',
        'continue to produce excessive yaw/convergence correction. Small selected-family residuals do not prove',
        'that the selected family is the one Adobe chose.',
        f"On the same {common['before']['n']} files with valid Full maps in BOTH versions, median error is "
        f"{common['before']['median']:.6f}% -> {common['candidate']['median']:.6f}%: the failure is not only denominator growth.",
        'Round 11 already showed that dropping the depth term alone does not explain the oracle gain.',
        '',
        '4. ALL-387 RAW: ORACLE CEILING AND IDENTITY WIN RATES',
        'Ungated candidate tables include identity for every no-map attempt; returned tables also include every',
        'quality rejection. Oracle includes all 26 diagnostic failures as identity, exactly as in Round 11.',
        'version           mode   maps accept   median%     p90%    W / L / T       wins/all% decisive%',
    ]
    for label,data in [('before',raw['before']),('final candidate',raw['candidate']),('oracle',s['oracle'])]:
        for mode in ['auto','full']:
            t=data[mode];v=t['candidate'];d=v['error']
            lines.append(f"{label:17} {mode:5} {t['fit']:5} {t['accepted']:6} {d['median']:10.6f} {d['p90']:9.6f}  "
                         f"{v['win']:3}/{v['lose']:3}/{v['tie']:3}       {v['win_all_pct']:8.3f} {v['win_decisive_pct']:9.3f}")
    lines += ['', 'Public returned outputs, original quality verdict:',
              'version           mode    median%     p90%      W / L / T       wins/all% decisive%']
    for label,data in [('before',raw['before']),('final candidate',raw['candidate'])]:
        for mode in ['auto','full']:
            v=data[mode]['returned'];d=v['error']
            lines.append(f"{label:17} {mode:5} {d['median']:10.6f} {d['p90']:9.6f}   {v['win']:3}/{v['lose']:3}/{v['tie']:3}       "
                         f"{v['win_all_pct']:8.3f} {v['win_decisive_pct']:9.3f}")
    lines += [
        '',
        'The candidate removes the Auto>Full all-attempt win-count inversion in THIS frozen ungated protocol:',
        'Full wins 37.47%, Auto 34.88%. But Full decisive wins are only 47.08%: it still loses to identity more',
        'often than it wins. The main target fails. More returned wins and a slightly lower returned median do',
        'not close the 74.55% ungated oracle gap. The baseline frozen public outputs already had Full above Auto',
        'on decisive wins; the supplied historical 40.7/86.1/25.8 rates are not reproduced under this protocol.',
        'All denominators and both gate policies are shown rather than importing those incomparable rates.',
        '',
        'On the 260 RAW validation files, returned medians (Auto/Full) change',
        '1.285320/3.306163% -> 1.315108/3.404585%; returned >5% counts change 17/87 -> 18/93.',
        'On the separate 228 decoded Adobe-abstained controls, baseline accepts none. The candidate accepts',
        'id2119129 Auto with 6.446840% error against identity. Full falls back. This is another nonregression failure.',
        'All-450 and calibration/validation/abstention summaries, paired counts, and tails: round12_summary.json.',
        '',
        '5. EXPLORATION — EVERY CALIBRATION VARIANT RETAINED',
        '150 fixed corrected calibration photos; Full/Auto medians include all failures as identity.',
        'variant                  Auto median%  Full median%',
    ]
    for name,z in s['calibration_variants'].items():
        lines.append(f"{name:24} {z['auto']['candidate']['error']['median']:12.6f} {z['full']['candidate']['error']['median']:13.6f}")
    lines += [
        'no_axis: remove the majority-orientation filter and fix the inconsistent pixel assignment tau; this is',
        'not a one-variable causal isolation. v1: dual-plane clustering/joint frame, sum score.',
        'v2: product family evidence and broader orthogonality score. v3: rank by newly explained support,',
        'distributed dual-plane probes.',
        'v4: add pose prior. v5: per-segment tangent seeds with iterative consensus expansion. v6: angular votes.',
        'v7: assign all segments in each orthogonal frame; broke exact synthetic recovery and was rejected.',
        'v8: v3 plus angular final assignment; best calibration Full median, but fails wrong-focal regression.',
        'v9: v8 plus focal-aware frame testing; fixes that regression. Both v8 and v9 were fully evaluated.',
        'Best initial all-RAW measured Full median was v8=3.667430%, still worse than 3.663819% baseline.',
        'The best 3.010925% calibration median is not substituted for all-387 performance. No positive all-RAW',
        'oracle-gap recovery was reached. These are exploratory comparisons, without independent-photo CIs.',
        '',
        '6. NONREGRESSION / FINAL FILE STATE',
        'Final focal-aware candidate: 105 tests pass (101 existing + 4 selector regression tests); initial v8 had',
        '100/101 existing tests pass, failing wrong-focal held-out recovery. Earlier v7 failed exact recovery.',
        'Both candidate and restored production have full 40-seed sweeps (4 scenes x 4 modes = 640 attempts),',
        '50-seed quality/tail audits (400 fixture + 700 held-out attempts), 280 held-out all-mode framed/scaled',
        'attempts, and 160 Adobe synthetic comparisons. Failures are retained in every output JSON.',
    ]
    for label,file in [('candidate','final_candidate'),('restored production','shipped')]:
        q=json.loads((E/f'round12_{file}_quality.json').read_text())
        for group in ['fixtures','heldout']:
            t=[t for scene in q['scenes'].values() if scene['group']==group for t in scene['trials']]
            z=np.array([x['returned_axis_deg'] for x in t])
            assert all(x['ok'] and x['gate_pass'] for x in t)
            lines.append(f"  {label:20} {group:8} n={len(t)} mean/max independent family error = {z.mean():.9f}/{z.max():.9f} deg")
    old_sweep=json.loads((E/'round12_shipped_sweep.json').read_text())
    new_sweep=json.loads((E/'round12_final_candidate_sweep.json').read_text())
    for scene in ['A_clean','B_strong','C_noisy','D_roll']:
        old=np.array([r['axis'] for r in old_sweep if r['scene']==scene and r['mode']=='full'])
        new=np.array([r['axis'] for r in new_sweep if r['scene']==scene and r['mode']=='full'])
        assert len(old)==len(new)==40
        lines.append(f"  Full 40-seed {scene:8} worst V/H: {old[:,0].max():.9f}/{old[:,1].max():.9f} -> {new[:,0].max():.9f}/{new[:,1].max():.9f} deg")
    base=json.loads((E/'round8.json').read_text())['measurements']
    new=json.loads((E/'round12_final_candidate_adobe.json').read_text())
    for prior in ['scene_focal','adobe_stored_focal']:
        a=np.array([r['final']['raw_rms_pct'] for r in base if r['prior']==prior])
        b=np.array([r['final']['raw_rms_pct'] for r in new if r['prior']==prior])
        lines.append(f"  Adobe {prior:19} n={len(a)} mean {a.mean():.9f}% -> {b.mean():.9f}%; "
                     f"max {a.max():.9f}% -> {b.max():.9f}%; {np.sum(b>a+1e-9)} individual attempts worsen")
    lines += [
        'Synthetic aggregate improvement does not override the real-photo Full tail and abstention regressions.',
        'Restored production: all 640 sweep maps, 280 held-out maps, and 160 Adobe scores exactly match immutable',
        'earlier results; all 105 tests pass. Source functions and all global policy constants are unchanged.',
        f"Production/before SHA256: {s['sha256']['port/upright_port.py']}",
        f"Experimental v9 SHA256:   {s['sha256']['evidence/round12_candidate_port.py']}",
        'Numerical matmul RuntimeWarnings remain visible in support-audit/replay logs; every scored successful',
        'candidate, returned score, and support-audit purity used here is finite. No warnings were hidden.',
        'No catalog write, native re-solve, photo modification, export, publish, detector or gate change occurred.',
        '',
        '7. REPRODUCE / ARTIFACTS',
        'From research/upright, Python 3.12, OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1:',
        '  ROUND12_PORT=evidence/round12_candidate_port.py python3.12 evidence/round12_probe.py --label final_candidate_all --workers 3',
        '  python3.12 evidence/round12_support_audit.py',
        '  ROUND12_PORT=evidence/round12_candidate_port.py python3.12 evidence/round12_validate.py STAGE --label final_candidate',
        '    STAGE = scope, sweep, heldout, quality, adobe',
        '  python3.12 -m pytest -q evidence/test_round12_selector.py',
        '  python3.12 evidence/round12_shipped_check.py STAGE',
        '    STAGE = sweep, heldout, quality, adobe',
        '  python3.12 -m pytest -q',
        '  python3.12 evidence/round12_analyze.py',
        '  python3.12 evidence/round12_checks.py',
        '  python3.12 evidence/write_round12_report.py',
        'To rerun the candidate against all 101 production tests without replacing the production file:',
        '  python3.12 evidence/round12_candidate_pytest.py',
        'round12_final_candidate_all.jsonl: all per-photo candidate, gate, identity, returned scores and maps.',
        'round12_support_audit.jsonl: before/after actual selected supports scored after solving.',
        'round12_*_quality.json, *_sweep.json, *_heldout.json, *_adobe.json: independent synthetic outputs.',
        'round12_summary.json: failure-retaining groups, oracle ceiling arithmetic, calibration variants, hashes.',
        'round12_checks.txt: final restoration, finite scores, scope, denominators and validation integrity checks.',
        'round12_before_port.py and round12_candidate*_port.py: immutable implementations; no unvalidated selector',
        'is substituted into port/upright_port.py. Historical Round-11 evidence remains untouched.',
    ]
    (E/'round12.txt').write_text('\n'.join(lines)+'\n')
    print('Wrote',E/'round12.txt',len(lines),'lines')

if __name__=='__main__':main()
