"""Build Round 3 evidence from retained per-attempt observations. MIT."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'
before=json.loads((E/'round3_before.json').read_text())
gated=json.loads((E/'round3_gate_only.json').read_text())
after=json.loads((E/'round3_final.json').read_text())
accuracy=json.loads((E/'round3_accuracy.json').read_text())
stress=json.loads((E/'round3_focal_stress.json').read_text())


def trials(result,group=None,seeds=50):
    return [t for s in result['scenes'].values() if group is None or s['group']==group
            for t in s['trials'][:seeds]]


def matrix(result,group=None,seeds=50):
    ts=trials(result,group,seeds)
    return [[sum(t['gate_pass']==p and (max(t['candidate_axis_deg'])>.1)==bad for t in ts)
             for bad in (False,True)] for p in (True,False)]


def pair(values):
    return '/'.join(f'{v:.6f}' for v in values)


def flag(t):
    return 'PASS' if t['gate_pass'] else 'FAIL'


old70=trials(before,'heldout',5);new70=trials(after,'heldout',5)
new700=trials(after,'heldout')
lines=[]
def write(text=''):lines.append(text)
write('ROUND 3 — PER-SEED TAIL, VP REFINEMENT AND GATE AUDIT — 2026-10-28')
write('='*72)
write('All changes inside research/upright/. Python 3.12; NumPy/OpenCV only; MIT.')
write('No raster, held-out scene, truth oracle, make_test_images.py or focal policy changed.')
write('Held-out RANSAC seeds 0..4 are the original protocol; 0..49 is an explicit extension.')
write('Generator noise seeds and all 14 camera cases remain fixed. No case/seed is omitted.')
write('')
write('1. EXACT REPRODUCTION AND RESULT')
write('The 0.113444312883 degree error is H09, RANSAC seed 0, V; H=0.081608153904.')
write('The other >0.1 is H09 seed 1: V=0.101511241478, H=0.070407524485.')
write('Round 2 prose incorrectly named seed 4. Its JSON was correct; historical files are untouched.')
write('The final H09 V/H errors for seeds 0..4 are:')
for t in after['scenes']['H09']['trials'][:5]:write(f"  seed {t['seed']}: {pair(t['returned_axis_deg'])} deg, {flag(t)}")
for label,ts in [('original held-out 70 before',old70),('original held-out 70 after',new70),
                 ('extended held-out 700 after',new700)]:
    scores=[t['returned_axis_deg'] for t in ts]
    write(f"{label}: mean family={np.mean(scores):.9f}, worst family={np.max(scores):.9f}, "
          f"rejected={sum(not t['ok'] for t in ts)}/{len(ts)}.")
write('Across all 1,100 final attempts (700 held-out, 400 fixture/focal), all pass and all errors <0.1.')
write('The 400 are four TIFFs x 50 plus 01_input at 1400/980/1200/1820 x 50; do not pool these')
write('with held-out headline accuracy. The four-TIFF historical comparison is kept separate.')
write('')
write('2. CAUSAL MECHANISM AND FIX')
write('Reproduced before production edits. Ranked hypotheses: stale post-reassignment VP;')
write('Round-1-style family selection; biased/ambiguous supports. The first is causal here.')
write('estimate_vanishing_points fits TLS on disjoint initial supports, then reassigns/expands')
write('inliers. It returned the newly expanded supports with the OLD subset-fitted VP.')
write('On H09 every original seed 0..4 selects the same 112 lateral and 189 vertical segments.')
write('Yet seed 0 lateral VP in normalized coordinates is (2.256882,-0.823849,1), versus')
write('(2.304001,-0.863254,1) at seed 2. The family identity and observed objective supports')
write('are unchanged. The depth support count varies 80..82. The horizon term consumes these')
write('stale lateral/depth VPs, forcing a seed-dependent roll/pitch/yaw compromise.')
write('This is not the Round 1 lateral/depth family swap and is not inherently ambiguous input.')
write('')
write('Controlled interventions, same raster/supports/focal, seed 0, independent world V/H:')
write('  original                         0.113444 / 0.081608')
write('  only disable horizon             0.011648 / 0.004884')
write('  TLS refit final supports         0.014163 / 0.005845')
write('  final robust selected refit      '+pair(after['scenes']['H09']['trials'][0]['returned_axis_deg']))
write('Full seed 0..4 interventions: round3_probe.txt; reproducible against round3_original_port.py.')
write('')
write('Production _base_solve now fixes the selected family identities, then refines the selected')
write('vertical/lateral/depth VPs on their final observed supports. _refine_vp uses centered')
write('isotropic coordinates and length-weighted Cauchy angular IRLS with the existing four-pixel')
write('angular scale (20 iterations maximum). Calibration, initialization and the enabled horizon')
write('term consume these refined VPs. The raw hypothesis remains only for selecting family identity.')
write('No world truth enters selection, refinement, calibration, optimization or the gate.')
write('')
write('Two rejected implementation trials explain why the fix is more than another TLS SVD:')
write('  - TLS refitting every VP creates angular-outlier bias on A/C and moves weak intersections.')
write('    round3_tls_trial.json retains every trial; 18 bad fixture/focal candidates, 2 falsely passed.')
write('  - Robustly refitting every unused VP before lateral selection can promote a weak unused')
write('    intersection at wrong focal. round3_all_vps_trial.json retains the trial: 30 bad fixture/')
write('    focal candidates, 26 falsely passed. Final refinement happens AFTER identity selection.')
write('These trials were rejected, not substituted for the final benchmark or hidden in means.')
write('')
write('3. WHAT THE QUALITY GATE MEASURES')
write('Old worst_scale is absolute detected-line median sine-angle / [4/(long_side-1)].')
write('It permitted up to 3 scale units: 0.672107 deg at H09 and 0.573445 deg at 01_input.')
write('That detects gross optimizer failures; it is not calibrated for a 0.1 deg systematic error.')
write('A smaller limit on the same absolute median would confuse detector noise with axis bias.')
write('')
write('01_input correction: the quoted ~0.13 deg is the absolute DETECTED-SUPPORT residual, not')
write('the independent world vertical error. Before, correct-prior seed 0 has world V/H')
write('0.013563/0.061668, support V median 0.134608, signed V median -0.022745.')
write('Final correct-prior seed 0:')
t=after['scenes']['01_input_1400']['trials'][0]
write('  world V/H='+pair(t['returned_axis_deg'])+'; support abs medians='+pair(t['quality']['family_median_deg']))
write('  signed medians='+pair(t['quality']['family_signed_median_deg'])+
      '; margin='+pair(t['quality']['family_median_margin_deg']))
write('  bias bounds='+pair(t['quality']['family_bias_bound_deg'])+
      f"; worst_scale={t['quality']['worst_scale']:.6f}.")
write('If 0.13 refers to another seed/oracle, it must be labeled separately; it is not the world')
write('error for this supplied fixture. Frozen legacy detector-inlier scores remain separate.')
write('')
write('The gate retains the old raw-scatter catastrophe check and ADDS an explicit systematic-bias')
write('budget: orient each output line toward its intended positive axis, compute the source-length-')
write('weighted signed pixel-angle median m and weighted MAD. n_eff=(sum lengths)^2/sum(lengths^2).')
write('Require |m| + 3.72*MAD/sqrt(n_eff) <= 0.1 deg for EACH family, before mode damping.')
write('3.72 approximates 2 x 1.253 x 1.483 for a median/MAD under independent Gaussian errors.')
write('This is a transparent support-consistency policy, NOT a statistical confidence interval or')
write('a universal world-error bound: edge fragments can correlate, scene axes can be mislabeled,')
write('and systematically biased evidence can fool an internally consistent fit. The current')
write('synthetic sample does not calibrate real-photo confidence. The .1 budget comes from the task;')
write('the four-pixel scale and median-margin constants are implementation choices, not Adobe rules.')
write('')
write('Both original bad H09 fits fail the NEW bias gate even if the fitter is not repaired:')
for t in gated['scenes']['H09']['trials'][:2]:
    write(f"  seed {t['seed']}: bound V/H={pair(t['quality']['family_bias_bound_deg'])}, "
          f"scatter_pass={t['quality']['scatter_passed']}, bias_pass={t['quality']['bias_passed']}.")
write('They return ok=False, identity, reason="poor undamped family alignment" when replayed')
write('through solve_ex. New tests also reject .11/.13 deg coherent bias across three aspects,')
write('and distinguish symmetric support scatter from the same supports with .13 deg bias.')
write('')
write('4. CONFUSION MATRICES — GATE vs INDEPENDENT ACTUAL ERROR >0.1 DEG')
write('BAD means max(V,H) of independent world-family medians >0.1, not the maximum individual line.')
write('Every candidate is scored, including rejections. Returned identity scores are stored separately;')
write('we do not call a rejected identity the rejected candidate. All 1,100 attempts have a candidate.')
write('Rows below are [gate pass, gate reject]; columns are [actual <=0.1, actual >0.1].')
for label,group,seeds in [('original held-out protocol','heldout',5),('extended held-out','heldout',50),
                         ('fixtures incl four 01 focal priors','fixtures',50),('all groups',None,50)]:
    write(label+':')
    for name,data in [('old fit / old gate',before),('old fit / NEW gate only',gated),('FINAL fit / NEW gate',after)]:
        m=matrix(data,group,seeds)
        write(f'  {name:26} pass {m[0][0]:4d} {m[0][1]:4d}; reject {m[1][0]:4d} {m[1][1]:4d}')
write('On the 700 held-out candidates before repair: old gate sensitivity to bad fits 0/11; new')
write('gate-only sensitivity 11/11, false rejects 0/689. On original 70: it rejects both bad fits.')
write('Gate-only on the older 400 fixture/focal fits conservatively rejects 40 actually-good fits,')
write('all on 01_input. Final robust refinement reduces their bias and all 400 pass. Thus the')
write('gate-only comparison has real positive examples; the final zero-bad matrix alone cannot')
write('establish sensitivity. This evaluation informed the fix and is not another untouched holdout.')
write('')
write('5. ADDITIONAL CONTROLS, CONTRACT CHANGES AND TESTS')
write('Wrong metadata remains real focal recovery, seed 0, true focal 1400 px:')
for f in (980,1200,1820):
    t=after['scenes']['01_input_'+str(f)]['trials'][0]
    write(f"  input {f}, source {t['source_focal_px']:.6f}, world V/H {pair(t['returned_axis_deg'])}.")
write('All 200 input-focal/seed combinations pass and remain within 3% focal and .1 deg axis error.')
write(f"Held-out +/-30% focal stress, seed 0, 28 attempts: mean {np.mean([t['axis'] for t in stress]):.9f}, "
      f"worst {np.max([t['axis'] for t in stress]):.9f}, rejected {sum(not t['ok'] for t in stress)}.")
write('')
write('Frozen OLD-detector controls (seed 0) are NOT silently removed:')
write('scene            gate  candidate V/H            returned V/H')
for n in ('A_clean','B_strong','C_noisy','D_roll','01_input_truth'):
    t=accuracy[n]['fixed_segments']
    write(f"{n:16} {'PASS' if t['ok'] else 'FAIL'}  {pair(t['candidate_axis_deg']):23} {pair(t['median_axis_deg'])}")
write('B_strong is a conservative false rejection: its candidate still meets the original .05 deg')
write('bound, but its noisy sparse supports do not meet the uncertainty budget. The existing energy-')
write('isolation assertion now checks this pre-gate candidate with the SAME .05 bound; benchmark')
write('returned-identity scores above remain visible. Old-support A/C/D calibration bias remains')
write('out of scope and is now rejected rather than confidently returned. No tolerance was widened.')
write('')
write('The old checkerboard smoke helper used arbitrary corner warping, requiring f^2 about -63k')
write('(bundled) / -67k (LSD) px^2 under centered square-pixel K. Its residual .46/.48 deg should')
write('fail the new gate. That exact old warp is retained in a rejection regression for BOTH')
write('detectors. The positive smoke helper now uses a real camera K*R*K^-1 and uses its pitch/yaw')
write('arguments (formerly ignored). Positive image-pipeline and detector-bypass assertions remain.')
write('No benchmark raster, renderer, truth or generating case was edited to get tests green.')
write('')
write('Verification:')
for filename in ('round3_pytest.txt','round3_unittest.txt'):
    content=(E/filename).read_text()
    write('  '+filename+': '+ ' | '.join(content.strip().splitlines()[-4:]))
write('New regressions: all 14 held-out scenes x 50 seeds, individual ok and max(V,H)<.1 plus')
write('global worst<.05; 01_input x 4 focals x 50 seeds; final-support VP stationarity; actual')
write('bad H09 optimizer replays; small-bias/scatter discrimination; impossible-camera rejection;')
write('confusion bookkeeping. Existing Round 1 seeds and Round 2 wrong-focal tests remain green.')
write('')
write('Reproduce from research/upright/:')
write('  python3.12 port/benchmark_accuracy.py --mode heldout --seeds 5 --output evidence/round3_heldout.json')
write('  python3.12 port/benchmark_quality.py --output evidence/round3_final.json')
write('  python3.12 evidence/round3_probe.py > evidence/round3_probe.txt')
write('  python3.12 port/benchmark_accuracy.py --output evidence/round3_accuracy.json')
write('  python3.12 -m pytest -q')
write("  python3.12 -m unittest discover -s port -p 'test*.py' -v")
write('  python3.12 evidence/write_round3_report.py')
write('Before implementation snapshot: round3_original_port.py. Gate-only keeps that old fitter')
write('with the final _alignment_quality, so all candidate scores match before within 1e-9.')
write('Per-attempt sources: round3_before.json, round3_gate_only.json, round3_final.json.')
write('')
write('6. FULL ORIGINAL HELD-OUT PER-SEED TABLE (70 attempts, seeds 0..4)')
write('Degrees. OLD and FINAL V/H use identical independent world oracles. BIAS is final gate bound.')
write('scene               seed       OLD V/H           FINAL V/H        old/gate-only/final    BIAS V/H')
for n,s in after['scenes'].items():
    if s['group']!='heldout':continue
    for t in s['trials'][:5]:
        seed=t['seed'];b=before['scenes'][n]['trials'][seed];g=gated['scenes'][n]['trials'][seed]
        write(f"{n:20} {seed:3d}  {pair(b['candidate_axis_deg']):19} {pair(t['returned_axis_deg']):19} "
              f"{flag(b)}/{flag(g)}/{flag(t)}  {pair(t['quality']['family_bias_bound_deg'])}")
write('')
write('7. EXTENDED FULL PER-SEED TABLE (all 1,100 attempts, seeds 0..49)')
write('Degrees. OLD means before candidate V/H; FINAL is actual returned V/H. No failed identities omitted.')
write('scene               seed       OLD V/H           FINAL V/H        old/gate-only/final    BIAS V/H')
for n,s in after['scenes'].items():
    for t in s['trials']:
        seed=t['seed'];b=before['scenes'][n]['trials'][seed];g=gated['scenes'][n]['trials'][seed]
        assert max(abs(np.array(b['candidate_axis_deg'])-g['candidate_axis_deg']))<1e-9
        write(f"{n:20} {seed:3d}  {pair(b['candidate_axis_deg']):19} {pair(t['returned_axis_deg']):19} "
              f"{flag(b)}/{flag(g)}/{flag(t)}  {pair(t['quality']['family_bias_bound_deg'])}")
(E/'round3.txt').write_text('\n'.join(lines)+'\n')
print('Wrote',E/'round3.txt',len(lines),'lines')
