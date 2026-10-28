"""Render the Round 4 report from the complete paired Adobe benchmark."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'evidence/round4_adobe.json').read_text())
rows = data['rows']
lines = []


def say(text=''):
    lines.append(text)


def select(scene=None, mode=None, version='after', prior='scene_focal', seed=None):
    return [r for r in rows if r['version'] == version and r['prior'] == prior
            and (scene is None or r['scene'] == scene)
            and (mode is None or r['mode'] == mode)
            and (seed is None or r['seed'] == seed)]


def mean(group, field, metric):
    return sum(r[field][metric] for r in group)/len(group)


say('ROUND 4 — FIDELITY TO ADOBE, NOT TO PERFECT RECTIFICATION — 2026-10-28')
say('='*88)
say('VERDICT: YES, on the four available Adobe scenes and the four cached automatic modes,')
say('the port is closer to Lightroom than a naive always-full rectification baseline.')
say('It is NOT an exact clone. Auto still over-corrects B_strong substantially, Full retains')
say("better geometry than Adobe, and Adobe's output framing is missing. This round improves")
say('shape agreement but REGRESSES raw mapping agreement relative to the starting port.')
say()
say('1. SCOPE, ORACLES AND METRICS')
say('All work stays inside research/upright/. No new dependency, native Adobe code, network,')
say('image, ground-truth scene, focal calibration policy, or quality threshold was introduced.')
say('Four adobe_truth/img_*.lua files exist. All four are scored; slots Auto=1, Full=2,')
say('Level=3, Vertical=4. Off/Guided identity cache slots are not automatic-solver evidence.')
say('No held-out raster has an Adobe cache here; the 70/700 geometric attempts cannot establish')
say('Adobe fidelity. A/C are related clean/noisy renderings; these are not four independent')
say('real photographs. Adobe is fixed cached output, not re-run for our RANSAC seeds.')
say('The actual port, port/upright_port.py, is evaluated. Existing validate_vs_adobe.py and')
say('compare_prototype_vs_adobe.py target upright_core.py, not this port; the former scores')
say('geometry. Their cache parsing/slot convention is reused, not their prototype results.')
say('Primary input: existing TIFFs, bundled detector, known generating focal 1600px, seeds')
say('0..4. Sensitivity: the SAME images/segments with each stored Adobe focal converted by')
say('the proven sensor-diagonal convention. This is an input sensitivity check, not tuning.')
say('Both versions receive the same segments/prior/seed; native H never enters the solver.')
say('80 returned mappings/version/prior = 4 scenes x 4 modes x 5 seeds; 320 paired rows total.')
say('Rejected identities are included. No scene, slot, or failing seed is omitted.')
say()
say('Raw: RMS Euclidean distance between Adobe and port output coordinates for the SAME')
say('625 source points on a 25x25 uniform grid including corners, in pixels, divided by')
say('source image diagonal and multiplied by 100. No crop/translation/zoom is removed.')
say('Shape: same output points after fitting ONE positive uniform output scale and a 2-D')
say('translation from the candidate to Adobe by least squares. Rotation, shear, aspect,')
say('and projective differences remain. This is explicitly a partial fidelity metric; it')
say('does not establish agreement with the whole stored H. Both metrics ignore H scalar gauge.')
say('This avoids comparing raw matrix coefficients, ground-truth residuals, or treating an')
say('area-normalized inverse composition as a complete Adobe homography agreement score.')
say('All sample mappings are finite and remain on one side of the projective horizon.')
say('Naive baseline: exact K(true) R(true) K(true)^-1, applied identically for EVERY mode,')
say('with centered K and no framing. It has perfect geometric knowledge. It is deliberately')
say('the requested full-rectification baseline, not a mode-aware competitor. The shape score')
say('removes arbitrary baseline output translation/zoom, so the verdict does not rest on its')
say('poor framing alone. All scenes/modes/seeds have equal weight; seeds are not independent')
say('Adobe observations. Small Full-only gains do not imply native Full has been recovered.')
say()
say('Stored native metadata (reported for provenance; source centers are not used by port):')
say('scene      f35_mm    diagonal_fpx    center_norm_x/y')
for name, scene in data['scenes'].items():
    m = scene['adobe_metadata']
    from math import hypot, sqrt
    focal = m['UprightFocalLength35mm']*hypot(scene['width'], scene['height'])/sqrt(36**2+24**2)
    say(f"{name:10} {m['UprightFocalLength35mm']:8.4f} {focal:14.6f}    "
        f"{m['UprightCenterNormX']:.6f}/{m['UprightCenterNormY']:.6f}")
say()
say('2. PER SCENE / MODE AGREEMENT WITH ADOBE H (mean over seeds 0..4)')
say('All distances below are percent of source image diagonal; lower is closer to Adobe.')
say('scene      mode        raw_before raw_after raw_naive | shape_before shape_after shape_naive')
for name in data['scenes']:
    for mode in ('auto', 'full', 'level', 'vertical'):
        old, new = select(name, mode, 'before'), select(name, mode)
        values = [mean(old,'agreement','raw_rms_pct'),mean(new,'agreement','raw_rms_pct'),
                  mean(new,'naive','raw_rms_pct'),mean(old,'agreement','shape_rms_pct'),
                  mean(new,'agreement','shape_rms_pct'),mean(new,'naive','shape_rms_pct')]
        say(f'{name:10} {mode:9} {values[0]:10.6f} {values[1]:9.6f} {values[2]:9.6f} | '
            f'{values[3]:12.6f} {values[4]:11.6f} {values[5]:11.6f}')
say()
say('Aggregate by mode (same equal-weight protocol):')
say('mode         raw_before raw_after raw_naive | shape_before shape_after shape_naive')
summary = data['summary']['scene_focal']
for mode in ('all','auto','full','level','vertical'):
    a, b = summary['before'][mode], summary['after'][mode]
    say(f"{mode:10} {a['raw_port']:10.6f} {b['raw_port']:9.6f} {b['raw_naive']:9.6f} | "
        f"{a['shape_port']:12.6f} {b['shape_port']:11.6f} {b['shape_naive']:11.6f}")
a, b = summary['before']['all'], summary['after']['all']
say(f"Against naive: raw RMS {b['raw_port']:.6f}% vs {b['raw_naive']:.6f}% "
    f"({100*(1-b['raw_port']/b['raw_naive']):.2f}% lower); shape {b['shape_port']:.6f}% "
    f"vs {b['shape_naive']:.6f}% ({100*(1-b['shape_port']/b['shape_naive']):.2f}% lower).")
say(f"Against starting port: raw WORSE by {100*(b['raw_port']/a['raw_port']-1):.2f}%; "
    f"shape better by {100*(1-b['shape_port']/a['shape_port']):.2f}%.")
say('The larger raw error is real: stronger pitch correction with the same centered K1')
say('moves the output farther from native framing. Shape alignment must not hide this.')
say()
say('3. DISAGREEMENT CLASSIFICATION — COMPONENTS, NOT ONE LABEL FOR AN ENTIRE H')
say('(a) implementation bug; (b) port geometrically better; (c) damping/mode semantics;')
say('(d) port worse/incomplete. A single H can have better pitch and worse roll/framing.')
say('Evidence used to distinguish them: native H is the fidelity target, independent world')
say('lines diagnose axis alignment, a world X/Z horizon diagnoses roll, and controlled changes')
say('separate damping from the full solve. A small homography distance alone proves none of these.')
say()
say('Shared findings:')
say('  (a) Different undamped objectives/gates per mode: Adobe-focal A/C accepted Full but')
say('      rejected Level (10/80 mode attempts). Fixed by using the unchanged Full objective')
say('      for every automatic mode before damping. All 80/80 now return a valid mapping.')
say('  (c) Auto used a branch-conditioned 14.58deg/3.56 curve on camera pitch, although the')
say('      requested pooled 13.33deg/2.64 curve is in visible-keystone units. Fixed the')
say('      constants and the x-axis, and converted physical convergence removal to pitch')
say('      rotation by a bounded scalar solve. Merely replacing constants on Euler pitch')
say('      made Auto under-correct still more; that is not the adopted change.')
say('  (d) All modes lack native destination camera/framing policy. The port fixes destination')
say('      principal point at 0.5/0.5; native maps include different translation, scale and')
say('      mild shape effects. Large raw errors are retained. No empirical per-scene recenter,')
say('      zoom, cache lookup, or invented Adobe crop policy was added. This was an explicit')
say('      pre-existing limitation, not evidence that the two-camera algebra itself is broken.')
say('  (c, residual) A pooled Auto response is not a native branch model. Coupled yaw/roll,')
say('      support distribution and focal branch behavior remain approximate. The requested')
say('      formula is implemented; exact recovery of these native choices is NOT claimed.')
say()
say('Per scene/mode diagnosis (the common framing class d applies to EVERY row):')
diagnoses = {
 'A_clean': {
  'auto': 'c fixed response/units; remaining V=b (slightly stronger), H=d (more axis bias).',
  'full': 'b: both families are much straighter than Adobe; retained, no weakening fit.',
  'level': 'a fixed gate at Adobe focal; b: native introduces ~0.50deg horizon tilt; port ~0.02deg.',
  'vertical': 'b in pitch: native leaves ~0.37deg V, port ~0.02deg; yaw intentionally retained.'},
 'B_strong': {
  'auto': 'c fixed response/units; branch mismatch persists: V=b but over-corrects as a clone; H=d.',
  'full': 'b: native V/H ~0.49/0.15deg, port ~0.05/0.01deg; retained.',
  'level': 'b: native introduces ~2.48deg horizon tilt; port ~0.02deg; retained.',
  'vertical': 'b: native V ~0.52deg, port ~0.05deg; similar intended uncorrected yaw.'},
 'C_noisy': {
  'auto': 'c fixed response/units; remaining V=b (slightly stronger), H=d (more axis bias).',
  'full': 'b: native V/H ~0.41/0.20deg, port ~0.02/0.03deg; retained.',
  'level': 'a fixed gate at Adobe focal; b: native horizon ~0.53deg, port ~0.02deg.',
  'vertical': 'b in pitch: native V ~0.42deg, port ~0.02deg; yaw intentionally retained.'},
 'D_roll': {
  'auto': 'c fixed response/units; remaining V=b, H=d; coupled yaw/roll is not native-equivalent.',
  'full': 'b: native V/H ~0.36/0.08deg, port ~0.014/0.001deg; retained.',
  'level': 'd: port horizon residual ~0.61deg vs Adobe ~0.25deg; fixed 0.87 roll is approximate.',
  'vertical': 'd: port V ~0.41deg vs Adobe ~0.36deg, despite straighter convergence; roll bias remains.'}}
for name, modes in diagnoses.items():
    for mode, diagnosis in modes.items():
        say(f'  {name:10} {mode:9} {diagnosis}')
say('Class (b) differences are documented differences from Lightroom, NOT clone successes.')
say('They are preserved as requested. Matching native weakness would require an explicitly')
say('evidenced native model; no hand-tuned weakening of Full/Vertical/Level was introduced.')
say('Unresolved class (c) behavior is stated openly: the evidence supports a fitted response,')
say('not a deterministic rule for native branch selection or coupled axes. No new branch rule')
say('is inferred from these four target matrices. Class (d) behavior is not modified this round.')
say()
say('Independent world axis residuals, degrees, seed 0, primary focal (diagnosis only):')
say('scene      mode          Adobe_V   Adobe_H     Port_V    Port_H   horizon_Adobe/Port')
for name in data['scenes']:
    for mode in ('auto','full','level','vertical'):
        r = select(name, mode, seed=0)[0]
        av, ah = r['adobe_world_axis_deg']; pv, ph = r['port_world_axis_deg']
        say(f"{name:10} {mode:9} {av:10.6f} {ah:9.6f} {pv:10.6f} {ph:9.6f}   "
            f"{r['adobe_horizon_error_deg']:.6f}/{r['port_horizon_error_deg']:.6f}")
say('Axis residuals use the original four independent world lines (two per family). Their')
say('symmetric absolute medians can hide roll errors in Level; the world horizon is therefore')
say('reported separately. These values never replace the native-H agreement metric.')
say()
say('4. AUTO: MEASURED CURVE AND ACTUAL ADOBE UNDER-CORRECTION')
say('Requested constants now used: sigma_V=13.33deg [9.94,22.72], L0_V=2.64, L0_H=76.73,')
say('sigma_yaw=11.64deg, sigma_pitch_gate=22.10deg, roll_auto=0.928. Vertical x-axis is')
say('P95-P5 signed vertical line direction spread in PIXEL angles, not Euler camera pitch.')
say('The response is lambda/(1+lambda), lambda=2.64*exp(-keystone^2/(2*13.33^2)).')
say('A bounded 32-step scalar inversion matches that removal fraction on selected observed')
say('vertical supports while retaining the separate yaw and roll schedules. If a target lies')
say('outside endpoint spreads, the pitch scale clips to [0,1]. This is our implementation,')
say('not recovered native source. Full/guided and explicit damping= reconstruction are retained.')
say('Observed edge fragments approximate the visible world lines used in the fit; sampling,')
say('noise, quantile ordering and native detector differences prevent exact transfer to world lines.')
say('The test checks physical removal on noiseless projected lines with both pitch signs.')
say()
say('seed 0: pitch = recovered undamped camera pitch; ks = observed keystone; requested =')
say('fitted removal on observed supports. Adobe/before/after fractions below instead use the')
say('SAME independent visible world verticals and P95-P5 angular spread, with the dense-sweep')
say('visibility filter (>=8 of 41 samples in frame). They are physical fractions, not H norms.')
say('scene       pitch_deg     ks_deg requested  Adobe_frac before_frac after_frac  after-minus-Adobe')
for name in data['scenes']:
    r, old = select(name,'auto',seed=0)[0],select(name,'auto','before',seed=0)[0]
    from math import exp
    request = 2.64*exp(-r['detected_keystone_deg']**2/(2*13.33**2))
    request /= 1+request
    say(f"{name:10} {r['rotation_deg'][0]:10.6f} {r['detected_keystone_deg']:10.6f} "
        f"{request:9.6f} {r['adobe_removed_fraction']:11.6f} {old['port_removed_fraction']:11.6f} "
        f"{r['port_removed_fraction']:10.6f} {r['port_removed_fraction']-r['adobe_removed_fraction']:+18.6f}")
say('Answer: starting Auto UNDER-corrected versus Adobe on all four scenes. Final Auto is')
say('still strongly damped relative to full rectification, but OVER-corrects versus Adobe:')
say('about 3 percentage points on A/C/D and about 30 points on B. It does NOT reproduce the')
say('complete Adobe Auto response. In particular B is NOT an implementation success just')
say('because its verticals are straighter. Its worse Auto shape score is retained in the mean.')
say('The constants alone cannot establish fidelity, and the direction was not assumed in advance.')
say()
say('Provenance cautions: evidence/sigma_v_precise.txt reports 80 exposed, 76 solved, 4 total')
say('abstentions. Its keystone fit C has sigma=13.33 but L0=2.474; L0=2.638 belongs to the')
say('camera-pitch fit A (sigma=22.26). The requested pair 13.33/2.64 is used verbatim, not')
say('misrepresented as one jointly fitted parameter pair. The pooled fit has ~0.210 fraction')
say('RMS error, a low-angle shoulder and high-angle branch bimodality. Its CI is not a per-image')
say('prediction bound. No new fit or bootstrap was run; the actual stored transforms remain')
say('the final target. The old branch-conditioned sigma evidence remains untouched.')
say('The corpus roll extraction and the port Euler decomposition also use different rotation')
say('conventions; independent axis scaling is consequently an approximation. Changing all mode')
say('axes to a different convention is not justified by the physical yaw sweep or these four')
say('matrices, and would change Level/Vertical semantics. This is an unresolved modeling limit.')
say()
say('5. STORED-FOCAL SENSITIVITY (mean over all seeds; no discarded identities)')
say('scene      mode         raw_after shape_after raw_naive shape_naive rejected_before/after')
for name in data['scenes']:
    for mode in ('auto','full','level','vertical'):
        old = select(name,mode,'before','adobe_stored_focal')
        new = select(name,mode,prior='adobe_stored_focal')
        say(f"{name:10} {mode:9} {mean(new,'agreement','raw_rms_pct'):10.6f} "
            f"{mean(new,'agreement','shape_rms_pct'):11.6f} {mean(new,'naive','raw_rms_pct'):9.6f} "
            f"{mean(new,'naive','shape_rms_pct'):11.6f} "
            f"{sum(not r['ok'] for r in old)}/5 -> {sum(not r['ok'] for r in new)}/5")
s = data['summary']['adobe_stored_focal']['after']['all']
say(f"Aggregate stored-focal raw={s['raw_port']:.6f}% vs naive={s['raw_naive']:.6f}%; "
    f"shape={s['shape_port']:.6f}% vs naive={s['shape_naive']:.6f}%; 0/80 rejected.")
say('Both focal protocols give the same baseline verdict. They are reported separately.')
say()
say('6. VERIFICATION AND REPRODUCTION')
say('Before: python3.12 -m pytest -q port -> 71 passed (round4_tests_before.txt).')
say('Regression red: three new tests fail before implementation (round4_regression_red.txt):')
say('wrong curve/units, missing physical-response conversion, and mode candidate inconsistency.')
say('Final: python3.12 -m pytest -q -> 75 passed (round4_pytest.txt), including original geometry,')
say('700 held-out full-mode seeds, 200 input/focal/seed combinations, gate discrimination,')
say('new physical damping checks, shared mode/gate checks, and independent mapping metric checks.')
say('The pre-existing pytest_asyncio configuration deprecation warning remains; no test fails.')
delta = max(abs(a-b) for r in select(mode='full') for a,b in zip(
            sum(r['H'],[]),sum(select(r['scene'],'full','before',seed=r['seed'])[0]['H'],[])))
assert delta == 0, delta
say(f'Primary Full homographies before/after: maximum coefficient change {delta:g} over all 20 pairs.')
say('The geometry fixes from Rounds 1–3 are preserved; no native weakness was injected into Full.')
say('To regenerate canonical data and report:')
say('  python3.12 port/benchmark_adobe.py --seeds 5')
say('  python3.12 evidence/write_round4_report.py')
say('  python3.12 -m pytest -q')
say('Artifacts: round4_original_port.py (exact starting port), round4_adobe.json (all 320 rows,')
say('returned and native matrices, per-seed scores and diagnostics), round4_adobe_run.txt,')
say('round4_before.json (initial seed-0 probe), regression/test logs, port/benchmark_adobe.py,')
say('port/test_round4.py, and updated port documentation. Historical round evidence is unchanged.')
say()
say('Final boundary: YES versus the specified naive baseline on the available caches. NO claim')
say('of exact native-H reproduction, complete Auto damping fidelity, improved raw fidelity this')
say('round, matched native crop/framing, held-out Adobe agreement, or real-photo generalization.')
(ROOT/'evidence/round4.txt').write_text('\n'.join(lines)+'\n')
print(f'wrote evidence/round4.txt ({len(lines)} lines)')
