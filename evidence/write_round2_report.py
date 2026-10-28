"""Assemble the Round 2 evidence from recorded runs; MIT."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
j=json.loads((ROOT/'evidence/round2_diagnosis.json').read_text())
h=json.loads((ROOT/'evidence/round2_heldout.json').read_text())
c=j['five_comparison'];four=j['four_comparison'];d=j['diagnosis']
fmt=lambda pair:'/ '.join(f'{x:.6f}' for x in pair)
parts=['''ROUND 2 — SOURCE FOCAL AND HELD-OUT RASTERS — 2026-10-28
=====================================================
Python 3.12.12, NumPy 2.2.6, OpenCV 4.13.0. MIT. No Adobe code or new
production dependency. All writes are inside research/upright/.
make_test_images.py is unchanged. Historical Round 1 evidence is preserved.

RESULT
------
01_input, Full, seed 0, supplied focal 1200 px (truth 1400):
  independent world V/H: 0.033602/0.513148 -> 0.025940/0.046189 deg.
  recovered source focal: 1426.188598 px; correct-prior result remains
  0.013563/0.061668 deg. Wrong-prior worst-family error is now lower than
  the current correct-prior worst-family error; the vertical component is
  still worse by about 0.0124 deg. This is not exact camera recovery.
  Real +/-30% supplied focals (980 and 1820) reach the same solution.

The historical legacy-inlier row is kept separate: its score becomes
0.232627/0.196949, from 0.189513/0.270213 (and original baseline
0.4366/0.223640). Its vertical component REGRESSES. With the correct 1400 px
prior that SAME legacy oracle gives 0.231317/0.207019, so the corrected
wrong-prior result is close to its right-prior result. Comparing a legacy
inlier score to the 0.013563/0.061668 world-truth score mixes oracles. We do
not claim the historical inlier row reaches 0.06 deg. The historical row's
1200 vs 1400 px is actually 14.3% low; tests also exercise true +/-30% errors.
''']
parts.append(f'''Separate held-out result, seeds 0..4 on each scene:
  14 scenes, 70 attempts, {h['failures']} returned-identity failures.
  mean family error {h['mean_family_deg']:.6f} deg;
  worst family error {h['worst_family_deg']:.6f} deg.
  matched five-fixture mean {c['fixture_mean_deg']:.6f} deg;
  held-out minus fixture mean {c['heldout_minus_fixture_deg']:+.6f} deg
  ({c['mean_change_percent']:+.2f}%). Mean accuracy does NOT degrade here.
  worst increases {c['fixture_worst_deg']:.6f} -> {h['worst_family_deg']:.6f},
  {h['worst_family_deg']/c['fixture_worst_deg']:.3f}x, or
  {h['worst_family_deg']-c['fixture_worst_deg']:+.6f} deg. Two held-out attempts
  exceed 0.1 deg. All errors are scored, including any returned identities.

The original four-TIFF headline is not pooled with the held-out scenes.
Its seed-0 mean remains 0.021186 deg. On matched seeds 0..4, its mean is
{four['fixture_mean_deg']:.6f}; held-out mean is {abs(four['mean_change_percent']):.2f}% lower,
but held-out worst is {h['worst_family_deg']/four['fixture_worst_deg']:.3f}x its worst.
The original five's seed-0 mean is 0.024472; held-out seed-0 mean is
0.013725. Seed averaging is explicit because H09 varies with RANSAC seed.

''')
parts.append('''1. REPRODUCTION AND CAUSAL DIAGNOSIS
------------------------------------
The regression was reproduced before production edits. The new real-raster
regression failed at all three supplied focals (980/1200/1820), with worst
world errors 1.263387/0.513148/0.604786 deg. See round2_regression_red.txt.
round2_before.txt reproduces the historical 0.189513/0.270213 legacy row.

Ranked hypotheses, tested independently:
  1. Fixed source focal makes the family rays nonorthogonal, preventing a
     rotation from aligning both axes. Prediction: calibrating K0 fixes it.
  2. Noisy, very distant fitted VPs bias direct focal recovery. Prediction:
     robust observed-line calibration improves on the closed-form TLS result.
  3. The extra depth/horizon term forces a bad compromise. Prediction: removing
     that term should eliminate the wrong-focal error if it is the root cause.

The supplied focal was NEVER an optimized variable. It built K0 once. The
existing w_f term only penalized (f1/f0 - 1), i.e. DESTINATION focal; it could
not change the SOURCE focal f0. Metadata [10,1000] and fallback 35 mm are
conversion policies, not a source-camera estimation strategy.

For centered square-pixel K0, perpendicular VPs a,b require
  f^2 = -[(a_x-cx*a_z)(b_x-cx*b_z)
          +(a_y-cy*a_z)(b_y-cy*b_z)] / (a_z*b_z).
With the wrong f, K0^-1 a and K0^-1 b are nonorthogonal. R preserves their dot
product. Sending both to the mutually perpendicular image axes is impossible
through R. Changing destination focal cannot repair source ray geometry.
''')
parts.append(f'''For 01_input the initial TLS pair gives {d['initial_geometry']['tls_pair_focal_px']:.6f} px,
versus truth 1400; at supplied 1200 its ray dot is
{d['initial_geometry']['prior_ray_dot']:.9f}. Robust observed-line calibration yields
1426.188598 instead. This probe supports hypothesis 1 and the need for robust
fitting in hypothesis 2; TLS focal recovery alone is noticeably biased.

Controlled interventions (same detected supports, seed 0, supplied 1200):
  intervention                   world V/H               legacy V/H
''')
for label in ('fixed_source','no_horizon','fixed_destination_centre','recalibrated'):
    r=d[label]
    parts.append(f"  {label:29} {fmt(r['truth_axis_deg']):24} {fmt(r['legacy_axis_deg'])}\n")
parts.append('''
The centre-drift interpretation was specifically falsified: free destination
centre hits (0.8,0.8); holding it at (0.5,0.5) gives the same angle errors to
~1e-9 and the same objective (0.000570138473719116). Translation changes no
line angle. Destination isotropic focal likewise changes no angle; its prior
pins it to f0. The centre is an unidentifiable direction in this angle-only
objective, and finite-difference roundoff drives it to a bound. There is no
meaningful picture-frame energy improvement from that drift. Widening bounds
would only allow a different arbitrary translation. The destination centre
now stays at (0.5,0.5) until an actual framing term exists. The two-camera
representation is retained; this policy changes framing, not axis scores.

Removing the horizon term leaves world H error 0.447499 deg; hypothesis 3 is
not the primary cause. Keeping the centre fixed alone also does not fix focal.
The source calibration reduces world H error to 0.046189 with the same
six-parameter destination objective and the horizon term still present.

2. IMPLEMENTATION AND LIMITS
----------------------------
Automatic modes now calibrate SOURCE focal first, using the existing Cauchy
pixel-angle residual on the selected vertical/lateral observed lines, with a
centered square-pixel camera and no free destination translation. Unknowns
are three rotation angles and log source focal. No world truth enters fitting,
family selection, acceptance or the public alignment-quality check.

The Jacobian's focal column is projected off its rotation columns. The
four-pixel angular scale divided by the remaining column norm measures focal
resolution. Near-frontal/infinite-VP geometry has little independent focal
information and retains the prior. Acceptance requires an interior fit,
log-focal resolution <0.1, and a focal discrepancy exceeding BOTH log(1.05)
and three times that resolution. These are implementation policies, not
Adobe rules, calibrated statistical intervals or guarantees about correctness.
The existing relative numerical search interval is not the metadata clamp.
Neither the metadata clamp nor fallback was widened or changed.

Small/plausible discrepancies retain the input focal: all 25 correct-prior
fixture attempts and all 70 correct-prior held-out attempts do so. The choice
was made on the focal repro before evaluating the held-out set. Guided and
shared_k comparison retain their original source-focal contracts. solve_ex
keeps focal_px as supplied and exposes source_focal_px plus calibration.
Damping overrides now rebuild with the recovered source focal.

A bad selected family or systematically biased supports can bias calibration.
IMPORTANT REGRESSION: on frozen ORIGINAL detector segments, the four-fixture
mean increases 0.106153 -> 0.193571 deg. Default source estimates become
1745.64/1600/1815.63/1816.07 px for A/B/C/D (truth 1600). Their default
before/after world V/H scores are retained below. This is not a universal
focal-calibration guarantee; the current detector and old supports behave
differently. No old-support row is dropped or rescored on optimizer inliers.

  scene        previous frozen V/H      current frozen V/H
''')
old=json.loads((ROOT/'evidence/round1_after.json').read_text())['accuracy']
for name in (*j['fixture_seeds'],):
    parts.append(f"  {name:17} {fmt(old[name]['fixed_segments']['median_axis_deg']):24} {fmt(j['accuracy'][name]['fixed_segments']['median_axis_deg'])}\n")
parts.append('''
3. HELD-OUT PROTOCOL — NEW GENERATOR, SEPARATE BENCHMARK
------------------------------------------------------
make_heldout_images.py uses NumPy camera rays intersecting planes and analytic
material sampling. It imports neither make_test_images.py nor upright_port.
This replaces PIL line strokes with soft tanh grout edges of different widths,
panel-dependent intensity, continuous world-coordinate texture and independent
sensor grain. It uses new world dimensions, view coverage and resolution.
Ground-truth continuous world centerlines are independently projected, clipped
in camera depth then to the image, and scored only if visible for >=40 px.
Vertical and lateral families are scored; side-wall depth lines are excluded
from axis residuals. Exact generating homographies rectify every world oracle
to <1e-8 deg, asserted by tests. Pixel samples and projected coordinates share
the integer-index convention.

Initial oracle QA found missing steep-pitch verticals due to rejecting lines
with one endpoint behind the camera, and a half-pixel coordinate mismatch.
Both GENERATOR/ORACLE errors were corrected; no solver parameter was tuned to
the held-out scores and no case was removed/replaced. The 12-camera matrix,
single-wall case and near-frontal case were specified before evaluation.
Final rendered images, truth NPZs and manifest are in heldout/. A contact sheet
is evidence/round2_contact.png. Benchmark regenerates in memory deterministically.

The source model still assumes orthogonal scene directions. Single wall here
means a non-three-family scene, as requested, with no depth family. This is
not an arbitrary oblique-wall test. Several telephoto frames naturally see
one wall only. No lens distortion, occluders, crop oracle or natural-photo
validation is included. Long regular soft-edged panels can be easier than the
old jagged strokes; the lower mean does not prove broad scene generalization.

Per-scene results below are means over seeds 0..4; maximum is over both
families and all five seeds. They are not folded into the fixture mean.

 scene                 size       mm pitch yaw  roll  V/H mean deg       max deg failures
''')
for name,row in h['scenes'].items():
    cam=row['camera'];tr=row['trials'];scores=np.array([t['median_axis_deg'] for t in tr])
    parts.append(f" {name:21} {cam['w']}x{cam['h']:<4} {cam['focal_mm']:3} {cam['pitch']:5} {cam['yaw']:4} {cam['roll']:5}  {fmt(scores.mean(axis=0)):20} {scores.max():.6f} {sum(not t['ok'] for t in tr)}/5\n")
parts.append('''
Worst: H09 (24 mm, 16:9, pitch18/yaw25/roll15), seed0,
0.113444 V / 0.081608 H deg. Seed4 also exceeds 0.1 V. These pass the
internal family-alignment quality gate; that gate is not a ground-truth bound.
H13 is the single wall; H14 is the near-frontal hard case (5/.25/.5 degrees).

Additional focal stress, SEPARATE from the correct-prior held-out benchmark:
all 14 images, supplied focal 0.7x and 1.3x truth, seed0 (28 attempts).
''')
s=[r for rs in j['heldout_focal_stress'].values() for r in rs]
for key in ('old','new'):
    scores=np.array([r[key+'_axis_deg'] for r in s])
    parts.append(f"  {key:3} policy: mean {scores.mean():.6f}, worst {scores.max():.6f}, failures {sum(not r[key+'_ok'] for r in s)}/28\n")
parts.append('''The old policy here holds source focal fixed and destination centre centered;
angles match the free-centre policy. All returned identities are included in
its mean. New calibration accepts 22/28 focal discrepancies; unresolved cases
keep the prior and still have small axis errors in these low-tilt views.
No production changes were made after observing this stress result.

4. REGRESSIONS AND VERIFICATION
-------------------------------
  python3.12 -m pytest -q
    63 passed (15.83s). Only the installed pytest-asyncio configuration warning.
  python3.12 -m unittest discover -s port -p 'test*.py' -v
    63 tests, OK (15.362s).

All original 50 tests plus Round 1 regressions remain green. The prior bad-
optimizer replay still tests its original six-parameter failure; its new
source-calibration stage runs normally instead of receiving that replay.
New coverage: real 01_input raster at 980/1200/1820 px with independent world
oracle, recovered focal, correct damping rebuild and fixed destination centre;
unobservable frontal calibration keeps prior; all 14 held-out exact oracles
and required camera coverage; actual detector/solver on all 14 held-out images;
all 28 held-out +/-30% metadata tests; explicitly injected rejections are counted
and scored as returned identity, rather than omitted from the mean.

Full Round 1 80-seed rerun: all successful, worst 0.050251 deg. All correct-
prior original five default image scores stay the same within numerical noise.
The source-calibration guard is inactive on those correct-prior fixture runs.

Full sensitivity run: 10 variants x 6 historical rows x 3 detector pipelines
= 180 attempts. Two least-squares fits are rejected. The current-detector
least-squares candidate mean regresses to 0.353392 deg; robust variants retain
their original current-detector means. Frozen-old-detector robust results also
regress, as stated above. All candidate and returned-identity scores, reasons
and quality data are retained in round2_sensitivity.json. Historical-control
replay explicitly reinstates the previous fixed-source/free-centre policy and
passes the original 1e-7 score comparison, rather than comparing new behavior
to an old immutable snapshot. No tolerance was widened.

5. REPRODUCE / FILES
--------------------
From research/upright/:
  python3.12 make_heldout_images.py
  python3.12 port/benchmark_accuracy.py --output evidence/round2_accuracy.json
  python3.12 port/benchmark_accuracy.py --mode heldout --seeds 5 --output evidence/round2_heldout.json
  python3.12 evidence/round2_diagnosis.py > evidence/round2_comparison.txt
  python3.12 port/benchmark_round1.py --output evidence/round2_round1_seeds.json
  python3.12 port/benchmark_sensitivity.py --output evidence/round2_sensitivity.json --record-failures
  python3.12 -m pytest -q
  python3.12 evidence/write_round2_report.py

Production: port/upright_port.py (source calibration, destination gauge,
source focal diagnostics, damping). Tests: port/test_round2.py plus the
existing optimizer replay in port/test_port.py. Benchmark: new --mode heldout
in port/benchmark_accuracy.py, same fixture default, plus a separate
01_input_wrong_focal_truth row. Historical inlier rows are retained. The
sensitivity control replay was updated for the new source policy. Docs:
port/README.md. All new files carry MIT designation or inherit port/LICENSE.

Source SHA256 values for the recorded run:
''')
for name,digest in j['hashes'].items():parts.append(f'  {name}: {digest}\n')
parts.append('''
Remaining finding: mean held-out accuracy is better on this synthetic set,
but the tail is 1.70x worse. Biased older detector supports regress under source
calibration. Those limits, rather than an Adobe-equivalence or real-photo claim,
are the extent of this evidence.
''')
(ROOT/'evidence/round2.txt').write_text(''.join(parts))
