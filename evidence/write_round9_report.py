"""Render Round 9 evidence without changing the frozen solver or policy."""
from pathlib import Path
import collections
import hashlib
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'port')]
from round9_probe import up,quad,zoom,geometry,json_default,self_check,boundary_controls
from benchmark_adobe import agreement

self_check()
held=json.loads((HERE/'round9_heldout.json').read_text())
sweep=json.loads((HERE/'round9_sweep.json').read_text())
photos=[json.loads(l) for l in (HERE/'round9_corpus.jsonl').read_text().splitlines()]
rows=[r for p in photos for r in p.get('rows',[])]
sample=json.loads((HERE/'round9_sample.json').read_text())
native=json.loads((HERE/'round9_native.json').read_text())
sampling=json.loads((HERE/'round9_sampling.json').read_text())
assert len(held)==280 and len(photos)==300 and len(rows)==600 and len(sample)==300
assert {p['image_id'] for p in photos}=={r['image_id'] for r in sample}
assert len({p['image_id'] for p in photos})==300
assert not any('decode_error' in p for p in photos)
assert all(r['metrics'][v]['raw_rms_pct'] is not None for r in rows for v in ('without','both'))
assert all(np.array_equal(r['H']['without'],np.eye(3)) and np.array_equal(r['H']['both'],np.eye(3)) for r in rows)
assert all(abs(r['metrics']['both']['raw_rms_pct']-r['native_identity_distance'])<1e-12 for r in rows)
assert not any(r['ok'] or r['without_ok'] for r in rows)
assert max(abs(r['decode']['aspect_error']) for r in rows)<.005


def stats(a):
    a=np.array(a,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        p90=float(np.percentile(a,90)),p99=float(np.percentile(a,99)),max=float(a.max()),
        over5=int(np.sum(a>5)))


summary={}
lines=[]
def say(s=''):
    lines.append(s)


say('ROUND 9 — FROZEN SCALE HOLDOUT + FIRST REAL-PHOTO END-TO-END TEST — 2026-10-28')
say('='*112)
say('VERDICT: the 14 synthetic scenes produce safe scales, but they do not validate Adobe scale matching.')
say('The frozen 1.10 switch has a real 9.09% zoom discontinuity and floating-point branch instability at equality.')
say('On 300 real photos / 600 Auto+Full attempts, the port accepts ZERO corrections. Every result is identity.')
say('Framing+scale therefore changes none of the real-photo scores. This is a failed real-photo clone result.')
say('A separate native-pose control also disproves broad scale-policy equivalence: 44/600 raw errors exceed 5%.')
say('No constant, damping, quality gate, detector or production function was changed or fitted in this round.')

say('\n1. PROTOCOL AND METRIC')
say('Primary raw clone distance: RMS pixel displacement over the existing fixed 25x25 source grid, including')
say('all four corners, divided by ORIGINAL source diagonal, times 100. No fitted translation, scale, rotation,')
say('crop or shape alignment is removed from the primary score. NumPy linear-interpolated percentiles.')
say('Each real photo contributes paired Auto and Full results, seed=0. Both option settings are fresh actual')
say('solve_ex calls, same supports/prior/seed; failed solves are retained as their returned identity. A photo')
say('is counted catastrophic if either mode exceeds 5%; an attempt is catastrophic if its raw score exceeds 5%.')
say('The heldout renderer is unchanged, and no four-scene data is loaded by this benchmark. All four automatic')
say('modes are tested on heldout seeds 0..4. Ground-truth quads are a policy-only control, never solver input.')
say('There are NO Adobe native transforms for these 14 scenes in this test; heldout safety and known-geometry')
say('agreement cannot establish that Adobe would select the same branch/zoom. No native GUI solve was run.')

say('\n2. FOURTEEN HELD-OUT SCENES — FROZEN FRAMING + SCALE')
say('Cells are ratio / C(cover) or I(inner) / applied scale, seed 0. Pose is pitch/yaw/roll in renderer degrees.')
say('All seeds are retained in round9_heldout.json; the branch is stable across all five seeds in every cell.')
say('scene                   pose       Auto                   Full                   Level                  Vertical')
for name in dict.fromkeys(r['scene'] for r in held):
    s=[r for r in held if r['scene']==name];c=s[0]['case']
    parts=[]
    for mode in ('auto','full','level','vertical'):
        m=[r for r in s if r['mode']==mode];g=m[0]['geometry']
        assert len({r['geometry']['branch'] for r in m})==1
        parts.append(f"{g['ratio']:.6f}/{g['branch'][0].upper()}/{g['scale']:.6f}")
    say(f"{name:<22} {c['pitch']:>2}/{c['yaw']:>5}/{c['roll']:<4} "+'  '.join(f'{p:<21}' for p in parts))
gs=[r['geometry'] for r in held]
say(f"Accepted {sum(r['ok'] for r in held)}/280; C={sum(g['branch']=='cover' for g in gs)}, I={sum(g['branch']=='inner' for g in gs)}.")
say(f"Scale range {min(g['scale'] for g in gs):.9f}..{max(g['scale'] for g in gs):.9f}; ratio range {min(g['ratio'] for g in gs):.9f}..{max(g['ratio'] for g in gs):.9f}.")
say(f"No nonfinite, inversion or wholly off-frame quad. Minimum intersection with unit output frame = {min(g['overlap'] for g in gs):.9f} square-frame units.")
max_seed=max(max(r['geometry']['scale'] for r in held if r['scene']==s and r['mode']==m)-min(r['geometry']['scale'] for r in held if r['scene']==s and r['mode']==m)
    for s in dict.fromkeys(r['scene'] for r in held) for m in ('auto','full','level','vertical'))
say(f'Max within-cell scale span over seeds = {max_seed:.9g}. Every cover-branch quad covers the full frame to numerical precision.')
say('Geometry interpretation: the switch responds to the gap between boundary fits, not monotonically to pitch')
say('or to a mode label. H01 and H14 select cover in every mode; H09 and H12 select inner in every mode.')
say('H05/H06/H11 select cover in Full but inner in Vertical: removing yaw changes the quad and candidate gap.')
say('H09/H12 even select inner for Level at high roll/wide aspect. Thus "Level always cover" is false, as expected')
say('for a geometry rule. Inner can still have scale >1 (e.g. H12); branch name does not specify zoom direction.')
say('Full vs independently rendered ground-truth quad, seed 0 (ratio / branch / scale):')
for name in dict.fromkeys(r['scene'] for r in held):
    r=next(r for r in held if r['scene']==name and r['mode']=='full' and r['seed']==0)
    g,t=r['geometry'],r['truth_geometry']
    say(f"  {name:<22} port {g['ratio']:.7f}/{g['branch']:<5}/{g['scale']:.7f}  truth {t['ratio']:.7f}/{t['branch']:<5}/{t['scale']:.7f}")
say('All 14 Full branches agree with the independent geometry control. This tests the solver-plus-rule geometry,')
say('not native Adobe behavior. Large pose alone does not predict branch: roll, yaw, aspect and focal interact.')

say('\n3. SYNTHETIC QUAD SWEEP AND FAILURE ENVELOPE')
rs=sweep['rows']
say('Deterministic shear, rotation, trapezoid and projective families; 1,001 deformation steps each; both axes')
say('and signs; x-aspect multipliers .5/1/2; uniform input sizes .1/1/10. Homographies solved in float64.')
say('Every quad is framed before scale. Only ratios in [1,1.5] (1e-10 endpoint tolerance) enter this band table.')
say('The full family sweep also checks outside-band cases for exceptions. Both candidate branches are explicitly')
say('forced for every band case, including the branch the frozen switch would not select.')
say(f"Band cases {len(rs):,}; cover {sum(r['geometry']['branch']=='cover' for r in rs):,}; inner {sum(r['geometry']['branch']=='inner' for r in rs):,}.")
say('family        cases     nonfinite  inverted  entirely-off-frame (selected OR either forced branch)')
for family in ('shear','rotation','trapezoid','projective'):
    ss=[r for r in rs if r['family']==family]
    counts=[sum(any((not c['finite']) if label=='finite' else c[label] for c in r['checks']) for r in ss) for label in ('finite','inversion','off_frame')]
    say(f'{family:<12} {len(ss):>7}     '+''.join(f'{v:>10}' for v in counts))
say(f"Band selected-scale range {min(r['geometry']['scale'] for r in rs):.9f}..{max(r['geometry']['scale'] for r in rs):.9f}.")
say('That range includes deliberately tiny/large input quads; absolute scale alone is not a safety criterion.')
say(f"Minimum selected frame intersection = {min(r['geometry']['overlap'] for r in rs):.9f}; no band case loses the frame.")
groups=collections.defaultdict(list)
for r in rs:
    groups[(r['family'],r['swap'],r['sign'],r['aspect'],r['size'])].append(r)
jumps=[];smooth=[]
for seq in groups.values():
    for a,b in zip(seq,seq[1:]):
        if abs(b['parameter']-a['parameter']-.001)>1e-10:
            continue
        change=abs(b['geometry']['scale']/a['geometry']['scale']-1)*100
        if a['geometry']['branch']!=b['geometry']['branch']:
            jumps.append(dict(family=a['family'],change=change,ratios=[a['geometry']['ratio'],b['geometry']['ratio']]))
        else:
            smooth.append(change)
say(f"Across adjacent .001 deformation steps: {len(jumps)} branch transitions; scale jump range {min(r['change'] for r in jumps):.5f}%..{max(r['change'] for r in jumps):.5f}%.")
say(f"Maximum same-branch adjacent relative scale change = {max(smooth):.6f}%. The large jumps are at the 1.10 switch.")
fail=sweep['family_failures']
say(f"Outside-band failure: {len(fail)}/144,144 family trials throw ZeroDivisionError at exactly 45-degree rotation")
say('(rotation parameter 1.0). The exact diamond has middle two x and y coordinates both equal to its centre,')
say('so inner=.5/0. It is a valid convex, positive-determinant, horizon-free map, not a malformed input:')
say('  H = [[1,-1,.5],[1,1,-.5],[0,0,1]]')
say('scale_output(H) raises ZeroDivisionError; solve_ex catches ValueError there, so this exception is unguarded.')
say('Its cover/inner ratio is undefined at exact collapse (limit 0), outside [1,1.5]. Some equivalent floating-point')
say('45-degree constructions have a tiny nonzero denominator and succeed: numerical representation controls the crash.')
stress=sweep['stress_accepted']
say(f"Additional 10,000 seeded random matrices: {len(stress)} accepted, {len(sweep['stress_rejections'])} rejected.")
say('Rejections: '+str(dict(collections.Counter(sweep['stress_rejections'])))+'.')
say('All accepted stress maps have finite, positive-area, overlapping output. No NaN/inf/inversion was returned.')
say('Scope: framing is required for this overlap guarantee. scale_output alone preserves the current bounds centre;')
say('a translated quad whose centre is off-frame can remain entirely off-frame if framing is omitted.')

say('\n4. THRESHOLD SENSITIVITY — NO REFITTING')
say('Exact shear family H=[[1,r-1,-(r-1)/2],[0,1,0],[0,0,1]], already centred: cover=r, inner=1.')
say('At r=1.10 the frozen rule selects 1.10; immediately above it selects 1.00: -9.090909% destination scale.')
say('The change persists as the geometry perturbation tends to zero. It is a discontinuity, not a large derivative.')
say('Raw cover-vs-inner jump uses the same 625-point metric on a 1200x800 frame. A changed threshold selects')
say('a different branch only for ratios between the old and new thresholds; elsewhere the output is unchanged.')
say('ratio   frozen branch/scale   cover-vs-inner raw gap %   threshold 1.09   threshold 1.11')
for r in sweep['sensitivity']:
    g=r['geometry'];a=next(t for t in r['thresholds'] if t['threshold']==1.09);b=next(t for t in r['thresholds'] if t['threshold']==1.11)
    say(f"{r['target_ratio']:.2f}    {g['branch']:<5}/{g['scale']:.6f}           {r['branch_gap_pct']:.6f}               {a['scale']:.6f}        {b['scale']:.6f}")
say('Boundary epsilon probe (actual frozen helper):')
for r in sweep['boundary']:
    say(f"  ratio target {r['target']:.10f}: computed {r['geometry']['ratio']:.12f}, {r['geometry']['branch']}, scale {r['geometry']['scale']:.12f}")
say('Equality is also numerically unstable. H with shear -0.1 selects scale 1.1; pre-framing the same already-centred')
say('H changes a rounding bit and selects scale 1.0. A translation of 1e-15 in x/y can do the same for shear +0.1.')
say('This breaks numerical commutation at the switch although the operations commute algebraically away from it.')
say('At displayed equality in the table, computed ratios can differ by one ulp; e.g. target 1.11 can remain inner')
say('at threshold 1.11. These are actual float64 branch decisions, not idealized decimal comparisons.')
say('Heldout sensitivity to changing ONLY the threshold; displacement is relative to frozen output, not an Adobe accuracy score:')
say('threshold    changed attempts/280    changed scene/mode cells/56    mean shift %    max shift %')
threshold_rows=[]
for threshold in np.round(np.arange(1.05,1.151,.01),2):
    shifts=[];changed=set()
    for r in held:
        g=r['geometry'];s=g['cover'] if g['cover']<=threshold*g['inner'] else g['inner']
        H=np.array(r['H']);alt=zoom(H,s/g['scale'])
        d=agreement(H,alt,r['case']['w'],r['case']['h'])['raw_rms_pct'];shifts.append(d)
        if abs(s-g['scale'])>1e-10:
            changed.add((r['scene'],r['mode'],r['seed']))
    tr=dict(threshold=float(threshold),attempts=len(changed),cells=len({(a,b) for a,b,_ in changed}),mean=float(np.mean(shifts)),max=float(max(shifts)))
    threshold_rows.append(tr)
    say(f"{threshold:.2f}                {len(changed):>3}                         {tr['cells']:>2}                 {tr['mean']:.6f}        {tr['max']:.6f}")
say('A threshold error of .01 can therefore change a few near-boundary outputs by about three source-diagonal percent;')
say('there is no evidence here for choosing a new constant. The full 11x11 synthetic ratio/threshold matrix is in round9_sweep.json.')

say('\n5. REAL CORPUS — SAMPLING, DECODING, AND RETURNED RESULTS')
say('Catalog opened with sqlite URI mode=ro. Observed snapshot counts (not assumed equal to the prompt snapshot):')
say(json.dumps(sampling,sort_keys=True))
say('Non-identity means max coefficient deviation from I >=1e-6 in BOTH slots 1 and 2, with positive catalog dimensions.')
say('Of 5,568 available qualifying paths, 114 project synthetic fixtures and 11 videos are excluded, leaving 5,443')
say('real unique paths. Uniform sample without replacement, Python Random(9009), sorted image IDs. No semantic')
say('filter for architecture, and no replacement after decoding or solving. All 300 decoded; no score was dropped.')
say('The source catalog reports 14,709 rows with the Auto key and 13,287 both-nonidentity rows, differing from 12,883')
say('in the request. This is an available-files sample, not a representative claim about the inaccessible photos.')
say('Sample formats: 255 RAW (111 DNG, 89 ARW, 42 NEF, 13 RAF), 33 TIFF, 12 JPEG.')
say('RAW: rawpy half-size, camera WB, no auto brightness, no orientation; LibRaw default crop mapped to visible RAW.')
say('TIFF: tifffile storage pixels, 16-bit RGB /257 to uint8. JPEG: Pillow RGB without EXIF transpose.')
say('A TIFF pilot revealed Pillow/libtiff silently applied orientation during decoding. All 33 TIFFs were re-run')
say('with storage-pixel decoding on the SAME sample, correcting this harness error before final reporting.')
say('Decoded aspect checks now all pass (<0.5% error). The first five photos were decoded and inspected; contact')
say('sheet round9_sample_contact.png verifies ordinary photographs and sensible viewing orientation.')
say('Detection at the default 1200-pixel maximum; supports then rescaled to original oriented catalog dimensions')
say('before solve_ex. Thus original-pixel quality tolerances and focal conversion are retained. Catalog orientation')
say('is applied once; output maps are conjugated back to sensor coordinates before scoring native transforms.')
say('The native Level off-diagonal ratio matches -(catalog h/w)^2, confirming the sensor-coordinate metric.')
say('Source prior is UprightFocalLength35mm through the existing diagonal conversion/metadata clamps; default')
say('source calibration is enabled. No native H enters the port. Manual perspective/crop settings are not composed')
say('into either score: target is the stored UprightTransform slot alone. Native source centres differ from .5;')
say('the port still uses its fixed centred-camera model. RAW rendering does not reproduce Adobe lens/tone processing.')
say('Distribution below is in percent of original source diagonal; all 600 attempts retained:')
say('mode     options          n       mean       median        p90         p99         max      >5%')
for mode in ('all','auto','full'):
    subset=[r for r in rows if mode=='all' or r['mode']==mode]
    summary[mode]={}
    for version in ('without','both'):
        d=stats([r['metrics'][version]['raw_rms_pct'] for r in subset]);summary[mode][version]=d
        say(f"{mode:<7} {version:<14} {d['n']:>4}   {d['mean']:>9.6f}   {d['median']:>9.6f}   {d['p90']:>9.6f}   {d['p99']:>9.6f}   {d['max']:>9.6f}   {d['over5']:>4}")
say('Acceptance with AND without framing+scale: 0/600. Reasons: 444 poor undamped family alignment,')
say('156 insufficient line structure (222 and 78 distinct photos respectively). Every returned H is exactly I.')
say('Framing-only scores are also identical. These are missed corrections / abstentions, not accepted bad warps.')
say('105/300 photos have a >5% error in at least one mode: 18 both, 86 Full only, 1 Auto only; 195 neither.')
say('The universal failure spans RAW, TIFF and JPEG, so RAW decoding alone cannot explain the lack of accepted solves.')
direct=json.loads((HERE/'round9_direct_jpeg.json').read_text())
assert len(direct)==24 and not any(r['ok'] for r in direct)
say('Independent pipeline check: all 12 JPEGs were also passed at full resolution directly to solve_ex(image=...),')
say('using its own default resize/detection path. All 24 mode attempts again rejected to exact identity. This')
say('confirms that the JPEG failure does not come from the benchmark support-rescaling seam or RAW processing.')
say(f"Numerical non-identity includes tiny native corrections: {sum(r['native_identity_distance']<.1 for r in rows)}/600 target slots differ from I by <.1% raw displacement.")
say('This run provides ZERO real-photo evidence that framing+scale improves an accepted end-to-end port correction.')

say('\n6. CATASTROPHIC ERROR CLUSTERS — DESCRIPTIVE, NOT A TRAINED DETECTOR')
say('group                         attempts    >5%     rate      median raw %')
cluster_rows=[]
for label,fun in (
    ('mode',lambda r:r['mode']),('reason',lambda r:r['reason']),
    ('orientation',lambda r:r['orientation']),
    ('format',lambda r:Path(r['path']).suffix.lower()),
    ('angle bound',lambda r:'hit' if r['rotation'] and max(abs(x) for x in r['rotation'])>=1.199999 else 'not hit'),
    ('supports',lambda r:'100-399' if r['segments']<400 else '>=400')):
    groups=collections.defaultdict(list)
    for r in rows:
        groups[fun(r)].append(r)
    for key,rr in sorted(groups.items()):
        a=[r['metrics']['both']['raw_rms_pct'] for r in rr];d=stats(a)
        cluster_rows.append(dict(group=label,key=key,**d))
        say(f"{(label+': '+key):<31} {len(rr):>5}     {d['over5']:>4}   {d['over5']/len(rr)*100:>6.2f}%      {d['median']:.6f}")
say('The strongest visible cluster is Full: 104/123 catastrophic attempts (84.6%), with a 34.7% failure rate')
say('versus Auto 6.3%. Rejected candidates hitting an Euler bound have a modestly higher rate; missing family')
say('structure also has a higher rate than the alignment-gate group. Catastrophes occur with hundreds of detected')
say('supports, so a simple low-segment-count rule does not isolate them. Tiny format/orientation groups are not reliable.')
say('Runtime ok=False flags every catastrophic result, but also every non-catastrophic result: 100% recall and')
say('20.5% precision at the attempt level, with zero usable corrections. It is not a selective catastrophe detector.')
say('Because output is I, catastrophe is EXACTLY Adobe-vs-I displacement >5%. This association with native correction')
say('strength is algebraic and requires Adobe truth; it is not a deployable detector or a newly fitted threshold.')
say('Worst 12 returned attempts (all identity):')
for r in sorted(rows,key=lambda r:r['metrics']['both']['raw_rms_pct'],reverse=True)[:12]:
    say(f"  {r['image_id']} {r['mode']:<4} {r['metrics']['both']['raw_rms_pct']:9.6f}%  {r['reason']}  {r['path']}")

say('\n7. ADDITIONAL CONTROL: FROZEN POLICY ON REAL NATIVE POSES (NOT END-TO-END PORT ACCURACY)')
say('Since all end-to-end solves reject, a separate scoring-only control isolates the policy using each native')
say('map. It repeats the Round 8 fixed-source-centre gauge: RQ(P H K0)=K1 R with positive focal diagonals;')
say('s_native=sqrt(fx1*fy1)/f0; divide the destination linear block by s_native, centre it, retain the same R,')
say('aspect and skew, then apply frozen frame_output/scale_output. No candidate scale is fitted to native points.')
say('This uses native rotation and is an oracle diagnostic, never fed to a port solve or mixed into the headline.')
nd=stats([r['raw'] for r in native]);summary['native_pose_control']=nd
say(f"600 slots: mean={nd['mean']:.6f}%, median={nd['median']:.6f}%, p90={nd['p90']:.6f}%, p99={nd['p99']:.6f}%, max={nd['max']:.6f}%; >5%={nd['over5']}.")
say(f"Framing law itself differs by >=.001% in {sum(r['native_framing_error']>=.001 for r in native)} slots; 41/44 >5% errors remain among slots obeying that framing law.")
say('61/600 slots are closer to the other boundary candidate, but none of the 44 >5% failures is brought below')
say('.01% by swapping candidates. Thus a slight threshold error alone cannot explain these real-pose scale failures.')
say('Some worst ratios are below 1 (outside the requested 1..1.5 band), exposing geometry absent from the old four scenes.')
say('This is direct evidence against treating the two-boundary family as a broadly verified Adobe scale clone.')
say('Native-pose worst 8 (actual catalog images, no filtering):')
for r in sorted(native,key=lambda r:r['raw'],reverse=True)[:8]:
    g=r['geometry']
    say(f"  {r['image_id']} {r['mode']:<4} raw={r['raw']:.6f}% ratio={g['ratio']:.6f} branch={g['branch']} native_s={r['native_scale']:.6f} predicted_s={g['scale']:.6f}")

say('\n8. VALIDATION / REPRODUCE / LIMITS')
say('Production SHA256 unchanged: '+hashlib.sha256((HERE.parent/'port/upright_port.py').read_bytes()).hexdigest())
say('Frozen sample SHA256: '+hashlib.sha256((HERE/'round9_sample.json').read_bytes()).hexdigest())
say('No tests, production code, mode constants, gates or renderer were edited. Harness self-checks verify orientation')
say('matrices on pixel centres, float64 homography reconstruction, clipping, a known ratio, frozen production hash,')
say('sample uniqueness, all 300 decoded, actual identity returns, and no dropped/invalid primary score.')
say('Full suite: 99 passed in 62.15s (evidence/round9_pytest.txt). Existing pytest-asyncio deprecation warning only.')
say('Commands from research/upright/:')
say('  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 evidence/round9_probe.py heldout')
say('  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 evidence/round9_probe.py sweep')
say('  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.12 evidence/round9_probe.py corpus --workers 3')
say('  OPENBLAS_NUM_THREADS=1 python3.12 evidence/round9_probe.py native')
say('  OPENBLAS_NUM_THREADS=1 python3.12 evidence/round9_probe.py direct_jpeg')
say('  python3.12 evidence/write_round9_report.py')
say('  python3.12 -m pytest -q')
say('The corpus command uses the frozen round9_sample.json. Optional stage sample re-queries mode=ro and redraws')
say('with the same seed; use the frozen file to reproduce this exact available-files cohort if the catalog changes.')
say('Per-attempt matrices/metadata/support diagnostics are in round9_corpus.jsonl; full native control in round9_native.json;')
say('full sweep/sensitivity in round9_sweep.json; every heldout mode/seed in round9_heldout.json. Before-TIFF-fix rows')
say('are preserved separately; no corrected row was selected based on score. All writes remain inside research/upright/.')
say('port/README.md now records the real-photo limitation and geometric failures; production code remains untouched.')
say('Limits: availability-biased personal-photo cohort; one seed per real photo; no matched Adobe preview/lens pipeline;')
say('no independently labeled real scene lines; no native solve of the 14 heldout images. These do not make the')
say('0/600 acceptance result acceptable, and the synthetic 0.587% headline must not be presented as real-photo accuracy.')

say('\n9. EVERY REAL PHOTO — FINAL RETURNED RAW DISTANCES (BOTH OPTIONS IDENTICAL)')
say('ID         Auto without/both %     Full without/both %    supports    reason (S=insufficient structure, Q=alignment)')
for p in photos:
    a,f=p['rows'];assert a['mode']=='auto' and f['mode']=='full'
    reason='S' if a['reason']=='insufficient line structure' else 'Q'
    say(f"{p['image_id']:<10} {a['metrics']['without']['raw_rms_pct']:9.6f}/{a['metrics']['both']['raw_rms_pct']:<9.6f}    {f['metrics']['without']['raw_rms_pct']:9.6f}/{f['metrics']['both']['raw_rms_pct']:<9.6f}     {a['segments']:>4}          {reason}")

summary.update(boundary_controls=boundary_controls(),heldout_attempts=len(held),heldout_accepted=sum(r['ok'] for r in held),
    real_accepted=0,real_photos=300,catastrophic_photos=105,threshold_sensitivity=threshold_rows,
    sweep_cases=len(rs),sweep_branch_transitions=jumps,sweep_family_failures=len(fail),clusters=cluster_rows)
(HERE/'round9_summary.json').write_text(json.dumps(summary,indent=2,default=json_default)+'\n')
(HERE/'round9.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:8]))
print('Wrote',HERE/'round9.txt',len(lines),'lines')
