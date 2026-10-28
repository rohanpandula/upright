"""Build Round 6 evidence from the completed offline probes and validation runs."""
from pathlib import Path
import hashlib
import json
import numpy as np

E = Path(__file__).resolve().parent
ROOT = E.parent
probe = json.loads((E/'round6_probe.json').read_text())
old = json.loads((E/'round5_adobe.json').read_text())
new = json.loads((E/'round6_adobe.json').read_text())
quality = json.loads((E/'round6_quality.json').read_text())
assert old == new
assert quality == json.loads((E/'round3_final.json').read_text())
for name in ('accuracy', 'sensitivity'):
    assert json.loads((E/f'round6_{name}.json').read_text()) == json.loads((E/f'round3_{name}.json').read_text())
assert (ROOT/'port/upright_port.py').read_bytes() == (E/'round5_original_port.py').read_bytes()
assert '75 passed' in (E/'round6_pytest.txt').read_text()
lines = []
def say(s=''):
    lines.extend(s.splitlines() or [''])
def mean(rows, field):
    return float(np.mean([r[field] for r in rows]))

say('''ROUND 6 — TEST ROTATION SPACE, DO NOT INFER IT FROM RAW DISTANCE — 2026-10-28
========================================================================================
VERDICT: The proposed structural mechanism is NOT CONFIRMED. The narrow camera/Euler
candidate's raw win is reproduced, but it does not establish Adobe's damping space.
Its A/C/D shape loss is real and its B convergence removal undershoots Adobe. The
requested camera22/Euler candidate is materially different and fails the requested
B raw <=25.38 target. No production behavior change is retained under "nothing may
regress". The original Auto over-correction is NOT fixed; it remains explicitly open.

1. WHAT WAS ACTUALLY TESTED
All work stays inside research/upright. No new dependency, quality threshold, focal
policy, line detector, native executable invocation, GUI change, catalog write,
target-H lookup in a solver, scene-specific rule, crop, zoom or fitted framing was
introduced. Native matrices enter OFFLINE SCORING ONLY. The catalog was SELECTed
with sqlite mode=ro to snapshot stored slots; Adobe was not re-run.

The primary reproduction uses the frozen Round 5 production source and identical
detected segments, source focal, random seed, shared undamped solve, K1 and gate.
Six Auto candidates are run on all four cached scenes, seeds 0..4, BOTH the 1600px
scene-focal protocol and stored-Adobe-focal sensitivity: 240 returned mappings.
All candidates preserve L0_V (also called LAMBDA0_V)=2.64, LAMBDA0_H=76.73,
SIGMA_YAW=11.64deg, SIGMA_PITCH_GATE=22.10deg, ROLL_AUTO=.928. No retuning.

New neutral names (historical evidence is not rewritten):
  keystone13_physical = current production: sigma=13.33deg on detected P95-P5
                        vertical direction spread, then invert physical removal.
  keystone13_euler    = same requested fraction, directly scale Euler rx.
  camera22_physical  = sigma=22.26deg on undamped camera rx, invert physical removal.
  camera22_euler     = sigma=22.26deg on undamped camera rx, directly scale Euler rx.
  camera13_euler     = sigma=13.33deg on undamped camera rx, directly scale Euler rx;
                       called wrong_sigma_camera_euler in Round 5.
  zero_pitch         = rx scale zero; keep yaw and roll schedules. Diagnostic only.
"Keystone units" means a spread of visible line directions in this report. It does
not mean camera pitch, sqrt(rx^2+ry^2), an Euler rotation ratio, or an output tilt.

The new harness reproduced all 100 matched Round 5 scene/seed/candidate rows to
1e-12 on raw, shape and physical removal. Reproduction is not an improvement claim.

2. CAUSAL RESULT: MOST OF THE RAW WIN IS NARROWER SIGMA, NOT EULER SCALING
Ranked hypotheses, stated before probes:
  (1) damping should act on rotation; (2) narrow sigma supplies most of the win;
  (3) raw framing dominates; (4) shape regression is an alignment artefact.
Controls below vary the argument and application space separately. Hypotheses
(2) and (3) explain the headline raw win; (4) is falsified. (1) is a plausible
approximation but is not identified as Adobe's implementation by these caches.

For B_strong, current -> camera22/physical -> camera22/Euler -> camera13/Euler:
  raw:       42.367262 -> 36.605380 -> 35.390363 -> 23.471712
  shape:      1.907538 ->  1.508652 ->  1.449169 ->  1.948337
  removed:    0.561334 ->  0.455881 ->  0.432040 ->  0.097243
Holding the camera22 argument fixed, Euler scaling buys only 1.215017 raw points.
Narrowing sigma then buys another 11.918651 points and REVERSES the shape gain.
The requested 22.26deg schedule does not reproduce the candidate's 23.47 raw score.
On A_clean the same final sigma change reduces raw 26.525377 ->19.376903, but
degrades shape .336177 ->.886020. A label alone cannot explain this difference.

Why raw falls: H=K1 R K0^-1 with the port's destination centre fixed at .5,.5.
For the source principal point, H maps K0*[0,0,1] to K1*R*[0,0,1]. Consequently
its output centre is c1 + (f1x*R02/R22, f1y*R12/R22). Reducing rx changes that
translation substantially. Adobe's framing keeps the image in quite a different
place. Suppressing rotation reduces this displacement without recovering Adobe's
line geometry. The zero-pitch control improves raw still further on all scenes,
and plain identity has raw 3.60/3.92/3.57/4.86 on A/B/C/D. Neither is an Auto clone.

The cache-independent centre-anchor control reverses the narrow candidate's raw
ranking on A/C/D: A 2.546712 ->2.979300; C 2.538438 ->2.952938;
D 3.573867 ->3.955942. It changes only output translation, and shape is identical
to 1e-12. This is a diagnostic, not a proposed Adobe framing model. A raw distance
win under the current arbitrary framing is insufficient evidence for damping space.

Why physical inversion and Euler scaling differ: visible convergence is a nonlinear
function of residual camera rotation, focal/FOV, support positions, yaw and roll.
Multiplying rx by f does not remove f of the line-direction spread. For seed-0 A,
current physical inversion gives rx scale .695046; camera22 Euler gives .618528;
camera13 Euler gives .404052. On B they are .577420, .447160 and .088833.
All rebuild H from the SAME undamped camera. This is a response-transfer change,
not a coordinate rename, and none establishes which transfer Adobe uses internally.

3. DOES THE NATIVE EVIDENCE ESTABLISH A POST-HOC EULER ROTATION SCHEDULE?
No. Section 3.8 establishes a rotation-dependent response, not the identity
"physical fraction = Euler fraction" or an internal sequence of solve-then-scale.
An adaptive objective keyed on camera rotation can yield the same observed curve.
The original dense-sweep observable is 1-mean_signed_line_tilt_after/before.
The comment equating it to corrected camera angle/true pitch is not an exact
geometric identity. Its fitted sigma=22.26deg and CI [16.67,39.58] belong to THAT
physical observable against known generating pitch. They are not an Euler-fit CI.

We therefore decomposed Auto AND Full from native stored slots, without a port fit:
  RQ(P H K0) = K1_general R, P=diag(w,h,1).
Every reconstruction and Euler round trip is asserted to 1e-12. K0 is fixed under
three explicit gauges: true 1600px/centred, stored focal/centred, and stored focal
plus stored principal point. K1 is general upper triangular. Its aspect/skew are
reported in JSON; RQ exactness is guaranteed for a nonsingular mapping, NOT proof
of Adobe's constrained square-pixel model or uniquely identified physical cameras.
Pitch ratios are stable across these gauges, but no internal stage is observable.

On the ORIGINAL 76 sweep rows, refitting the original physical response recovers
L0=2.638426, sigma=22.261080deg, RMS=.209669. Fitting native Auto/Full Euler rx
ratios instead gives L0=3.475329, sigma=20.214849deg, RMS=.209016 using true K0;
stored K0 gives 3.489633 /20.045877deg, RMS=.204443. These fits are descriptive,
not proposed new production constants, and no inherited CI is claimed.
With L0=2.64 fixed, camera22 RMS is .209666 for the physical observable and
.215914 for Euler ratios (true K0). Camera13 is worse: .302749 and .319757.
Thus the full sweep does not favor interpreting the physical fit as an Euler law.
Comparable noisy residuals cannot identify Adobe's internal damping space.

Another limitation of the older corpus claim: corpus_lambda_curve.py extracts
polar(K^-1 H K), assuming one camera. That is not the two-camera R when K1 != K0.
For A Auto, stored-K0 RQ pitch is -14.0015deg; same-K polar gives -6.2620deg.
For B Auto they are -9.3565deg versus -4.3956deg. The older corpus's "keystone"
bins are hypot(rx,ry) from that polar extraction, not visible convergence. It is
useful behavioral evidence but cannot prove direct Euler scaling or its units.

Native mixed-scene rotations also expose coupling the fixed schedules do not match:
B's RQ Auto rotation (true K0) is (-9.428620,6.011923,2.616407)deg. Port Euler22
applies (-15.310270,11.681419,.025958)deg; narrow Euler13 applies
(-3.041537,11.681419,.025958)deg. Reducing pitch cannot repair this yaw/roll gap.

Native rotations under all fixed-source gauges:
scene       gauge                       Auto rx      Auto ry      Auto rz  Auto/Full rx
''')
for r in probe['native']['records'][:4]:
    for g in ('true_K0', 'stored_focal_centred_K0', 'stored_K0'):
        a, f = r['slots']['1'][g], r['slots']['2'][g]
        say(f"{r['name']:11} {g:27} " + ' '.join(f'{x:12.6f}' for x in a['euler_deg']) + f" {a['euler_deg'][0]/f['euler_deg'][0]:13.6f}")

say('''
4. SHAPE TRADE-OFF: REAL, NOT A SCALE/TRANSLATION FIT ARTEFACT
Raw and legacy shape use the fixed 625-point grid INCLUDING corners, in percent
of SOURCE diagonal. Shape fits ONE positive uniform output scale and translation;
rotation, anisotropy, shear and perspective remain. They measure different losses.
The shape fit is asymmetric in pixel units, so we add a symmetric statistic:
  A=a-mean(a), B=b-mean(b), rho=<A,B>/(||A|| ||B||)
  symmetric_shape=100*sqrt(1-rho^2).
This is invariant to translating/scaling EITHER mapping and to swapping them;
it does not fit away roll. It is percent of target centred RMS radius, NOT percent
of source diagonal. Both definitions agree on the regression. Reverse-alignment
and centre-anchor invariance are asserted in the harness.
We also measure RMS differences between the SAME independent world verticals'
output directions (degrees, invariant to output scale/translation), and repeat
legacy shape on a 13x13 grid restricted to the middle 60% of the source frame.

A camera13 regression: legacy shape .251394 ->.886020; symmetric .766388 ->2.701081;
vertical-direction RMS .478563 ->.875128deg; interior shape .142735 ->.456754.
It is neither an asymmetric-scale artefact nor a corner-only effect. The smaller
camera22 A loss also survives symmetric/interior scoring, although its vertical
direction RMS improves: it trades other geometry components against verticals.

FULL CANDIDATE TABLE, means across all five paired seeds. No rejected row omitted.
raw/shape/interior/anchor/translation-only: percent SOURCE diagonal;
sym: percent centred radius; Vdiff: degrees; removed: visible convergence fraction.
camera13 is the user's highlighted Round 5 candidate, named neutrally here.
''')
for prior in ('scene_focal', 'adobe_stored_focal'):
    say(prior)
    say('scene       candidate                 raw       shape       sym     Vdiff   removed  interior    anchor  transOnly rejects')
    for name in ('A_clean', 'B_strong', 'C_noisy', 'D_roll'):
        for label in ('keystone13_physical', 'keystone13_euler', 'camera22_physical', 'camera22_euler', 'camera13_euler', 'zero_pitch'):
            rs = [r for r in probe['candidates']['rows'] if r['scene']==name and r['prior']==prior and r['variant']==label]
            fields = ('raw_rms_pct', 'shape_rms_pct', 'symmetric_shape_pct', 'vertical_direction_rms_deg', 'removed', 'interior_shape_rms_pct', 'centre_anchored_raw_rms_pct', 'translation_only_rms_pct')
            say(f'{name:11} {label:23} '+ ' '.join(f'{mean(rs,k):9.6f}' for k in fields) + f" {sum(not r['ok'] for r in rs)}/5")
    say()

say('''5. DOES ADOBE ACTUALLY RETURN NEAR-IDENTITY AT 34 DEG?
Sometimes on the pure-pitch sweep, not generally, and not literally on B_strong.
B_strong's true pitch is 34.377468deg, yaw 12.032114deg; it is not pure-pitch V34.
Its STORED normalized Auto matrix, H22 normalized to one, is:
  [[ .93423568, -.07437278, -.00734109],
   [ .04450717,  .93884879, -.04398755],
   [-.08502842, -.09447370, 1.00000000]].
||H-I||F=.183631484; max|H-I|=.094473700. Grid identity RMS=3.919597% diagonal;
Full ||H-I||F=.586201244, grid RMS=7.110379%. Auto removes .264781 of convergence;
it is weak compared with Full, but it is not an identity mapping or a .097 branch.

At EXACT 34deg pure pitch, the stored Auto magnitudes are .469787451, .449204997,
.453606813, .053050414. Three seeds remove .729-.763 of convergence, one .110.
The native response is a mixture, not "high pitch => near-identity". At 38/40deg
the weak branch can be genuinely tiny. The full native table below reports all
80 sweep exposures plus the four mixed scenes, rather than selecting weak seeds.

Cache-cohort audit: the current catalog contains all 80 slots; the saved original
fit had 76. The additional names are V38S3,V38S4,V40S3,V40S4. All 76 pre-existing
physical measurements are reproduced to 1e-12, so the old CI/fit remains attributed
to that original cohort. No new native solve was performed. The extra current
V40S3 has a BAD Full solution (negative removed convergence); it is retained in
the native table and never counted as proof that Full always succeeds. The old
"four missing/abstained" description is historical, not current catalog state.

FULL NATIVE CACHE TABLE. norm = ||H/H22-I||F in normalized coordinates, NOT pixels;
raw = 625-point RMS displacement from identity in percent source diagonal;
removed = independent P95-P5 convergence removal, including negative values.
name          f35mm  Auto norm Full norm   Auto raw  Full raw Auto removed Full removed
''')
for r in probe['native']['records']:
    a, f = r['slots']['1'], r['slots']['2']
    say(f"{r['name']:13} {r['metadata']['UprightFocalLength35mm']:7.4f} {a['frobenius_H_minus_I']:9.6f} {f['frobenius_H_minus_I']:9.6f} {a['identity_raw_rms_pct']:10.6f} {f['identity_raw_rms_pct']:9.6f} {a['physical_removed']:12.6f} {f['physical_removed']:12.6f}")

say('''
Native dense-sweep response fits, ALL targets explicitly distinguished:
cohort       target                        L0    sigma(deg)    fitRMS  fixed13RMS fixed22RMS
''')
for cohort, fits in probe['native']['fits'].items():
    for target, f in fits.items():
        say(f"{cohort:12} {target:27} {f['L0']:9.6f} {f['sigma_deg']:12.6f} {f['rms']:9.6f} {f['fixed_curves']['13.33']['rms']:10.6f} {f['fixed_curves']['22.26']['rms']:10.6f}")

say('''
6. WHICH WOULD SHIP, AND THE MILD-REGIME CONTROL
Ship the retained current implementation under the user's no-regression constraint.
This is a conservative scope decision, not a claim that current Auto is correct.
Camera13 wins raw but loses shape on ALL FOUR scenes, misses B's measured native
convergence in the opposite direction, and suppresses the three strong 34deg slots.
Camera22/Euler improves B raw and shape and improves aggregate Auto shape, but
regresses A/C/D shape and fails B raw <=25.38. We cannot call that "nothing regressed".
The condition for task 2 (confirmed mechanism) was not met: no falsely attributed
22.26deg Euler CI or renaming of a physical fit as a rotation fit is shipped.
LAMBDA0_V/L0_V, horizontal/gate/roll constants remain untouched.

At the mild 15deg/6deg non-Manhattan/control scenes, narrow camera13/Euler moves
AWAY from the earlier native .83-.91 ideal-shape response. Below is the report's
own sim_residual fraction (max residual after output similarity against the best
of Full and pitch-only ideal, normalized to that ideal's identity residual).
This is neither raw/shape RMS nor physical convergence removal. Means of seeds
0..4; all 150 candidate attempts are retained, no rejection.
scene                ks13phys ks13Euler cam22phys cam22Euler cam13Euler zeroPitch
''')
for name in sorted({r['scene'] for r in probe['candidates']['mild']}):
    say(f'{name:21} '+ ' '.join(f"{mean([r for r in probe['candidates']['mild'] if r['scene']==name and r['variant']==label], 'ideal_fraction'):9.6f}" for label in ('keystone13_physical', 'keystone13_euler', 'camera22_physical', 'camera22_euler', 'camera13_euler', 'zero_pitch')))

say('''
7. FINAL PRODUCTION VALIDATION — ACTUAL ROUND 5 FINAL -> ROUND 6 FINAL
The benchmark's internal "before" field is the OLD Round 4 START snapshot. It is
not this round's baseline. Tables below pair Round 5 AFTER to Round 6 AFTER.
All 320 benchmark records/summaries, including the historical control rows, are
exactly equal between round5_adobe.json and round6_adobe.json. All 160 production
H matrices/scores across four modes, four scenes, five seeds, two focal protocols
are exactly equal; zero production rejections. No mode can regress unnoticed.

FULL PER-SCENE/MODE FINAL TABLE, five seeds each; raw and shape are both reported.
prior               scene       mode       raw before raw after shape before shape after
''')
for prior in ('scene_focal', 'adobe_stored_focal'):
    for name in ('A_clean', 'B_strong', 'C_noisy', 'D_roll'):
        for mode in ('auto', 'full', 'level', 'vertical'):
            a = [r for r in old['rows'] if r['version']=='after' and r['prior']==prior and r['scene']==name and r['mode']==mode]
            b = [r for r in new['rows'] if r['version']=='after' and r['prior']==prior and r['scene']==name and r['mode']==mode]
            values = [np.mean([r['agreement'][field] for r in rs]) for field in ('raw_rms_pct', 'shape_rms_pct') for rs in (a, b)]
            say(f'{prior:19} {name:11} {mode:9} '+' '.join(f'{v:11.6f}' for v in values))

say('\nAggregate final raw / shape, unchanged:')
for prior in ('scene_focal', 'adobe_stored_focal'):
    for mode, r in new['summary'][prior]['after'].items():
        say(f"  {prior:19} {mode:9} {r['raw_port']:10.6f} / {r['shape_port']:9.6f}, rejects {r['rejected']}/{r['attempts']}")

say('''
RE-RUNS AND TAILS:
  python3.12 -m pytest -q: 75 passed (52.07s). Existing pytest-asyncio warning only.
  This includes all 700 held-out Full seeds, 200 focal/seed combinations for
  01_input, gate discrimination, Round 4 response invariants and guided tests.
  Separate benchmark_quality.py: 400 fixture/focal +700 held-out Full attempts;
  every candidate passes the gate AND independent V/H error <=.1deg; no dropped
  identities, false passes, false rejections or missing fits.
''')
for group in ('fixtures', 'heldout'):
    trials = [t for s in quality['scenes'].values() if s['group']==group for t in s['trials']]
    scores = np.array([t['returned_axis_deg'] for t in trials]); mx = scores.max(1)
    say(f"  {group:8}: n={len(trials)}, mean family={scores.mean():.9f}deg, max-family p95={np.percentile(mx,95):.9f}deg, p99={np.percentile(mx,99):.9f}deg, worst={scores.max():.9f}deg.")
say('''  The complete quality JSON equals the prior round3_final.json exactly.
  benchmark_accuracy fixtures and benchmark_sensitivity (loss/scale/support
  variants, record-failures enabled) both equal their Round 3 JSON exactly.
  Known bad sensitivity/control variants remain reported; they are not production
  regressions and are not hidden by claiming every exploratory variant passes.
  The diagnostic script asserts exact RQ/Euler reconstruction, symmetric/reverse
  shape agreement, centre-anchor invariance and all 76 historical sweep values.
''')
say('Production SHA256: '+hashlib.sha256((ROOT/'port/upright_port.py').read_bytes()).hexdigest())
say('''
8. ARTIFACTS, REPRODUCTION AND LIMITS
round6_native_slots.json: immutable offline snapshot of current 80 stored slot sets
and only the needed metadata, exported via sqlite mode=ro. No live catalog needed
to reproduce the analysis. Four mixed scenes still come from adobe_truth/img_*.lua.
round6_probe.py / round6_probe.json: all 240 candidate scene mappings, 150 mild
attempts, 84 native records with matrices/magnitudes/source gauges, 76/80 fits.
round6_adobe.json: full 320 historical/current benchmark records, including every
seed, both priors, all four modes and every returned H. Final-current rows=160.
round6_quality.json: all 1100 independent-scoring attempts, including every tail.
round6_accuracy.json, round6_sensitivity.json, round6_pytest.txt: separate checks.

Reproduce from research/upright (python3.12 has the installed dependencies):
  python3.12 evidence/round6_probe.py > evidence/round6_probe_run.txt
  python3.12 port/benchmark_adobe.py --seeds 5 --output evidence/round6_adobe.json
  python3.12 port/benchmark_quality.py --seeds 50 --output evidence/round6_quality.json
  python3.12 port/benchmark_accuracy.py --output evidence/round6_accuracy.json
  python3.12 port/benchmark_sensitivity.py --record-failures --output evidence/round6_sensitivity.json
  python3.12 -m pytest -q
  python3.12 evidence/write_round6_report.py

Limits: synthetic related scenes only; no held-out Adobe homographies. RQ is a
fixed-source-gauge diagnostic with general destination intrinsics, not unique
recovery of Adobe's internal camera or proof of post-hoc Euler scaling. The new
rotation fits are descriptive and have no newly bootstrapped CI. No native solver
was re-run, no real-photo validation is claimed, and the output framing/branch/
coupled-rotation gaps remain. We would need independent evidence discriminating
those mechanisms before replacing the current response under a no-regression rule.
''')

(E/'round6.txt').write_text('\n'.join(lines)+'\n')
print(f'Wrote evidence/round6.txt ({len(lines)} lines)')
