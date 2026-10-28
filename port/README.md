# upright_port

A reference implementation of the geometry behind Lightroom-style **Upright**
perspective correction: a clean-room rewrite derived from a behavioural study of
the shipped solver, written to be lifted into a permissively licensed project.

* Pure numpy + OpenCV, one file, no build step.
* MIT licensed. No Adobe source code; only the algebra, re-derived, plus
  numbers read out of or fitted against the shipped binary.
* Constants carry provenance tags (`PROVEN` / `MEASURED` / `PAPER` / `CHOICE`) with
  an evidence reference, so you can see which parts are facts, which are fits,
  and which are straight from the paper.

**Current real-photo result: still a poor Adobe clone.** On 450 photographs Adobe
corrects, 900 Auto/Full attempts with framing and scale have median **2.076%**,
p90 **7.979%**, p99 **17.419%**, maximum **81.962%** raw source-diagonal error.
Counts above 1/5/20% are **628/179/5**. Only **80/900** corrections are accepted;
the rest return identity. The calibrated gate fixes resolution-dependent
rejection, but broad useful coverage remains unsolved. Protocol, untouched
validation, abstention controls and confusion matrices: [round10.txt](../evidence/round10.txt).

Round 11 isolates a support-selection problem: on 387 RAW files, replacing the
render with the embedded JPEG does not improve the paired result. Adobe-assisted
axis labels on the same RAW detections reduce ungated Full median error from
3.664% to 0.932%, including 26 identity fallbacks. This is an oracle diagnostic,
not a shipped improvement. Full-size decoding and generic radial correction do
not resolve the gap; production loading and the gate are unchanged. Protocol,
controls and limitations: [round11.txt](../evidence/round11.txt).

Round 12 tested VP-based support assignment, joint Manhattan frames, explicit
roll handling, and the two-part ambiguity margin. The final candidate did not
close the oracle gap: Full RAW median was 3.672% versus 3.664%, with worse
tails. It remains an experiment; production is unchanged. Results and retained
candidate: [round12.txt](../evidence/round12.txt).

```python
import upright_port as up

H = up.solve(segments, w, h, focal_px, mode="full")     # your own segments
H = up.solve(image=gray, focal_px=3200.0, mode="auto")  # bundled detector
H = up.solve(segments, w, h, focal_px, mode="auto", framing=True)
H = up.solve(segments, w, h, focal_px, mode="auto", framing=True, scaling=True)
```

Output framing is opt-in. `frame_output(H)` also applies it to an existing solve.
It centres a 90% inner / 10% outer blend of the corrected quad's bounds, matching
the translation of all 16 active cached Adobe slots to stored precision.
`solve_ex(..., framing=True)` retains the original mapping as `H_unframed`.
Framing preserves shape, rotation, damping and output scale.
Derivation and framing-only scores: [round7.txt](../evidence/round7.txt).

Destination scaling is a separate opt-in: `scale_output(H)` or `scaling=True`.
It computes the scale covering the original frame and the scale containing the
corrected quad's inner bounds. If their ratio is at most 1.10 it uses the cover
scale; otherwise it uses the inner-bounds scale. This measured rule uses only
the quad, without mode-specific constants. It preserves the 90/10 bounds centre,
so it commutes with framing and leaves all line angles and damping unchanged.
`H_unscaled` retains the mapping before scaling. Both options default to False.
`PerspectiveScale=100` is the manual setting, separate from this implicit sizing.
The rule closely reproduces the old cached synthetic scales, but Round 9 found
a 9.09% zoom discontinuity at the switch and division by zero on an exact diamond.
In Round 9, all 600 Auto/Full attempts on 300 real photos rejected to identity, so
framing and scaling improved none of those results. A separate native-pose
control exceeded 5% raw error in 44/600 slots. See [round9.txt](../evidence/round9.txt).
Derivation, correlations, rejected hypotheses and raw clone distances:
[round8.txt](../evidence/round8.txt).

`H` acts on **normalized** coordinates (`x` in `[0, 1]` across the width, `y` in
`[0, 1]` down the height). To apply it to an image:

```python
N = np.diag([1.0 / w, 1.0 / h, 1.0])
H_px = np.linalg.inv(N) @ H @ N
out = cv2.warpPerspective(img, H_px, (w, h))
```

## Model

```
H = s * K1 * R * K0^-1          K1 != K0
```

`K0` is the **source** camera: its focal starts from EXIF (the shipped solver
stores the *source* focal, see `evidence/k0_identification.txt`). Our automatic
modes now check that prior against robust perpendicular-family geometry.
`K1` is the
**destination** camera the solver invents: a free focal and a free principal
point in the model. Our angle-only objective cannot identify its principal
point, so automatic solves hold it at the image centre. `R` is the rectifying
rotation, `s` an irrelevant scale.

Keeping the two cameras distinct is the whole point. Collapsing the model to the
textbook `K * R * K^-1` reproduces a measured Adobe homography only to a median
orthogonality residual of **3.2e-1**; the two-camera form reaches **5.7e-16**,
i.e. machine zero, over 29 real solves (`evidence/orthogonality.txt`,
`evidence/model_exactness.txt`). The destination principal point alone accounts
for most of the difference.

## Modes

| mode | rotation applied | notes |
| --- | --- | --- |
| `off` | none | returns identity |
| `level` | `rz * 0.87` | roll only |
| `vertical` | `rx`, `rz * 0.87` | pitch + roll |
| `full` | `rx`, `ry`, `rz` | everything |
| `auto` | measured saturating response approximation | remaining native differences below |
| `guided` | solved from 4 constraints | 4 hard constraints, residual ~1e-13 deg |

`auto` is the default. Its requested vertical convergence removal is
`lambda / (lambda + 1)`, with `lambda = L0_V * exp(-keystone^2 / 2 sigma_v^2)`.
Keystone is the P95-P5 signed direction spread of detected vertical supports in
pixel angles; it is not camera pitch. A bounded scalar solve converts that
physical removal fraction into a pitch rotation scale. This support-based
transfer is our approximation to the visible world lines used in the measured
fit. Yaw uses `LAMBDA0_H * exp(-yaw^2 / 2 sigma_yaw^2)` gated on pitch, and
Euler roll uses 0.928. All automatic modes now share Full's undamped candidate
and quality decision. Explicit `damping=` still overrides Euler scales.

These are approximate mode responses, not an exact Adobe clone. On the four
cached Adobe scenes, Round 4 Auto removes about 67–69% of visible convergence
on A/C/D versus Adobe's 64–65%, and 56% on B versus Adobe's 26%. A smooth
pooled curve does not reproduce Adobe's branch behavior or its coupled yaw/roll.
Full deliberately retains the port's more accurate rectification; Adobe leaves
about 0.36–0.49° vertical error on these scenes. Output framing and scaling are
opt-in (`framing=True`, `scaling=True`); without them the emitted map is centred
and unframed and scores ~33% against Adobe. See [`round4.txt`](../evidence/round4.txt) for per-scene/mode
mapping scores, classifications, before/after regressions and the limited
verdict against a naive full-rectification baseline. Reproduce with
`python3.12 port/benchmark_adobe.py --seeds 5` from `research/upright`.

`guided` is the highest-value mode: it needs no energy model and no prior. Pass
exactly four segments (at least one vertical and one horizontal); it solves the
four residuals `(rx, ry, rz, log f1)` to zero. Any other count raises
`ValueError`.

## Provenance of constants

| constant | value | tag | source |
| --- | --- | --- | --- |
| `SIGMA_V` | 13.33 deg (CI 9.94-22.72), keystone units | MEASURED | `evidence/sigma_v_precise.txt`, fit C; replaces the old 14.58° camera-pitch branch fit |
| `L0_V` | 2.64 | MEASURED/requested | Round 4 requested value; precise evidence fit A reports 2.638, fit C reports 2.474, so this pair is an approximation |
| `LAMBDA0_H` | 76.73 | MEASURED | `evidence/sweep2_resolution.txt` |
| `SIGMA_YAW` | 11.64 deg | MEASURED | Round 4 requested value; `evidence/sweep2_resolution.txt` |
| `SIGMA_PITCH_GATE` | 22.10 deg | MEASURED | Round 4 requested constant; `evidence/sweep2_resolution.txt` fits 22.05, `evidence/abstention.txt` does not contain it |
| `ROLL_AUTO` | 0.928 | MEASURED | `evidence/lambda_curve.txt`, n = 5854 |
| `ROLL_LEVEL` | 0.87 | MEASURED | `evidence/round4.txt`; described there as fixed and approximate |
| `SENSOR_DIAGONAL_MM` | 43.266615305567875 | PROVEN | `evidence/focal_prior.txt`; full-frame 36x24 **diagonal** |
| `DEFAULT_FOCAL_MM` | 35.0 | PROVEN | `evidence/focal_prior.txt`; fallback in four places, including the failure path |
| `FOCAL_CLAMP_MM` | (10, 1000) | PROVEN | `evidence/focal_prior.txt`; applies to metadata only |
| `COS_22_5` | 0.9238795042037964 | PROVEN | `evidence/detector.txt`; 8-direction gradient quantisation |
| `RANSAC_ITERS` / `MIN_INLIERS` / `TAU` | 2000 / 3 / sqrt(3) | PROVEN / CHOICE | `evidence/vp_clustering.txt`; 3 is proven, 2000 is a candidate cap and sqrt(3) a vote-kernel tau |
| `MAX_VP_CANDIDATES` | 9 | PROVEN | `evidence/vp_clustering.txt` |
| `NFA_P` / test count | 1/8 / `maxdim**4` | PROVEN | `evidence/detector_corrections.txt`; actual binomial tail |
| `REFINE_GAP_PX` / `REFINE_DISTANCE_PX` | 10 / 1.0 | PROVEN | `evidence/detector_corrections.txt`; the reported 1/2048 was a float-width error |
| `MERGE_PERP_PX` / `MERGE_GAP_PX` | 6 / 10 | PROVEN | `evidence/detector_corrections.txt`; containment slack, not gap bridging |
| Cauchy loss scale | 1.0 | CHOICE | port's own; not in the binary |
| Quality-gate scale | `max(0.1 deg, family scatter)` | CHOICE | round 10; replaced a `4/(maxdim-1)` pixel rule that failed at 8K |
| Framing blend | 10% outer / 90% inner | MEASURED | `evidence/round7.txt`; fitted on the same 16 slots it is scored on |
| Scale switch | cover, else fit inside above 1.10 | MEASURED / CHOICE | `evidence/round8.txt`; the 1.10 coefficient is measured, the branch form is the port's choice |
| `SIGMA_V_PAPER` / `SIGMA_H_PAPER` | `pi/12`, `pi/15` | PAPER | DOI [10.1109/CVPR.2012.6247761](https://doi.org/10.1109/CVPR.2012.6247761); kept only for A/B |

Focal conversion is by **diagonal**, not by the short edge:

```
focal_px = f35 * hypot(w, h) / 43.266615305567875
```

The usual `f35 / 36 * max(w, h)` is wrong by 1.8 % on 3:2 and by 4.8 % on 4:3.
Use `up.focal_px_from_35mm(f35, w, h)` and `up.focal_35mm_from_px(focal_px, w, h)`
rather than doing it by hand. A focal you obtained *geometrically* is taken as
given and never clamped; only metadata goes through `metadata_focal_px`.

## Detector

`detect_segments` stays separate: `solve(segments=...)` accepts any `(N, 4)`
pixel array, including OpenCV LSD output. It now uses:

* Eight fixed direction scans and the recovered `cos(22.5°)` angular test.
* Float32 `round(4 * log(maxdim) / ln(8))` support filtering in the smoothed
  pass; no fixed eight-pixel minimum or density cutoff.
* The actual NFA: `maxdim**4 * P(Binomial(n, 1/8) >= aligned_count)`, evaluated
  in log space without integer overflow or probability underflow.
* Subpixel edge localization and bounded TLS support splitting; refinement call
  values `(10, 1.0)` and near-containment absorption `(6, 10.0)`.
* An unsmoothed fallback when the first pass finds nothing, using NFA alone.
  Each pass requires NFA < 1; the combined nominal false-alarm budget is <= 2.
  This is our policy, not a recovered native detector stage.

The raw bytes contradict several interpretations in `detector.txt`: the refine
argument is float32 **1.0**, perpendicular merge distance is **6 px**, the NFA
really is binomial, and native bin rounding is **floor**. See
[`detector_corrections.txt`](../evidence/detector_corrections.txt) for addresses.
The requested nearest-round bin rule stays for compatibility with the original
34 tests. Gradient preprocessing, tracing and support refinement remain
re-derived; complete pixel equivalence is **not established**. The angular
literals `1/3600, 1/8100, 1/900`, `(4/(maxdim-1))**2` and `0.16` belong to a
separate camera parameter block; unknown roles are not invented as detector gates.
The pre-existing VP estimator remains an approximation of the stage described in
`vp_resolution.txt`.

## Tests

```
cd port
python3.12 -m unittest discover -p 'test*.py' -v
```

77 tests across the port suite. They cover the diagonal focal conversion against known
values, exact round-trip recovery with `damping=1`, guided convergence of both
families below 0.01 deg, the two-camera vs single-camera orthogonality gap,
mode ordering, coordinate conventions, detector behaviour and constant
provenance. Requires numpy and OpenCV; `python3.12` is the interpreter that has
them on this machine.

## Accuracy

**The 0.587% figure below is synthetic-only.** It is four 1400x1000 CG scenes, and
98% of it comes from the framing/scale policy, not the solver: oracle geometry under
the same policy scores 0.598%. On real photographs the port still returns identity
for roughly 90% of frames and loses to doing nothing on the rest — see
`../evidence/real_photos.txt` and `../evidence/round10.txt`. Do not quote 0.587% as
real-photo accuracy.

Full-mode median vertical / horizontal axis residual, measured on independent
world geometry in the bundled rasters (degrees):

| scene | before | after |
| --- | --- | --- |
| A_clean | 0.406 / 0.163 | 0.018 / 0.014 |
| B_strong | 0.390 / 0.187 | 0.050 / 0.031 |
| C_noisy | 0.212 / 0.403 | 0.020 / 0.022 |
| D_roll | 0.090 / 0.339 | 0.014 / 0.001 |
| 01_input, true focal | 0.192 / 0.162 | 0.014 / 0.062 |

The four-TIFF mean family score drops **0.274° → 0.021° (92%)**. With the old
detector segments frozen, changing only the energy lowers it to **0.106°**;
B_strong alone drops to **0.040° / 0.028°**. The objective now scores observed
lines in pixel angles, uses source-length weights and a Cauchy loss, and removes
the false fork penalty on arbitrary two-line intersections.

The adversarial review in [`port_review.txt`](../evidence/port_review.txt)
tests loss/scale/length sensitivity and lists report corrections. Robust variants
retain 89.1–92.3% improvement; least squares retains 80.2%. These are five related
synthetic fixtures. The seed-4 failure was traced to ranking horizontal VPs
from the image origin; ranking camera-centered rays fixes it to 0.018° / 0.012°.
The round-1 regression covers D_roll seeds 0–49 and A/B/C seeds 0–9; all stay
below 0.1° in both families. The rejected detector trial's seed-9 failure is
also fixed. See [`round1.txt`](../evidence/round1.txt) for the causal probes and
every before/after seed score. This remains a synthetic-scene result.

`solve_ex` reports `quality` for automatic modes before mode damping. It retains
raw length-weighted absolute median pixel-angle scatter `a`, with a fixed **1°**
cap calibrated on real photographs. For each family, signed bias plus its support
margin, `b = abs(median) + 3.72 * weighted_MAD / sqrt(effective_count)`, must
satisfy **`b <= 0.1°` or `b < a`**. The strict inequality rejects coherent bias
even with zero MAD; noisy photographic evidence gets a scatter-based budget.
Both tests are independent of pixel resolution for fixed normalized geometry.
The margin is a consistency heuristic, not a calibrated confidence interval;
correlated or systematically biased supports can defeat it. `worst_scale`
remains a legacy four-pixel diagnostic and does not control acceptance.
Diagnostics expose medians, margins, bounds, per-family bias limits and both
decisions. Failure returns `ok=False`, identity, and
`reason="poor undamped family alignment"`. Guided has separate guide diagnostics.
Calibration and limitations: [round10.txt](../evidence/round10.txt).

Reproduce with `python3.12 benchmark_accuracy.py`. Frozen original segments and
homographies are in `../evidence/accuracy_baseline_segments.npz`; before/after
JSON and the full protocol are in
[`accuracy_gap.txt`](../evidence/accuracy_gap.txt). These are axis deviations,
not the report's two-line convergence metric. `01_input.png` is synthetic and
has a known 1400px focal; the historical 1200px-focal/inlier diagnostic is also
preserved separately; it measures frozen detector inliers rather than world truth.

Round 2 adds a source-focal check before the destination solve. It fits the
observed vertical/lateral supports with a centered square-pixel source camera,
using the existing robust pixel-angle loss. A discrepancy must exceed both 5%
and three times a profiled four-pixel focal resolution, and the fit must be
interior and informative. These are our policies, not recovered Adobe rules or
statistical confidence bounds. Guided and `shared_k` retain the supplied focal.
`solve_ex` preserves the input in `focal_px`, reports the used source focal in
`source_focal_px`, and exposes `calibration` diagnostics. Metadata fallback and
clamps remain 35 mm and [10, 1000] mm.

In Round 2, `01_input` supplied 980/1200/1820 px recovered about 1426 px (truth 1400),
giving 0.026° / 0.046° on independent world lines, versus 0.014° / 0.062° for
correct metadata. The historical frozen-inlier score remains a separate oracle.

The independent NumPy ray/plane renderer is `../make_heldout_images.py`;
`make_test_images.py` is unchanged. Run the second benchmark separately:

```
python3.12 benchmark_accuracy.py --mode heldout --seeds 5 --output ../evidence/round3_heldout.json
```

Fourteen scenes cover every 3:2/4:3/16:9 × 24/35/50/85 mm combination, plus a
single wall and near-frontal control, with soft grout edges, panel shading and
grain. In Round 2 on matched seeds 0–4, the five-fixture mean is 0.023712°, and the held-out
mean is 0.008777° (63.0% lower). The worst family error increases from 0.066923°
to 0.113444° (1.70×); two of 70 held-out attempts exceed 0.1°, none are rejected.
These results are separate from the original four-TIFF headline mean. The
regular panels may be easier than the fixtures; this is not a real-photo test.
The separate held-out ±30% focal stress check gives 0.015706° mean, worst
0.111461°, zero rejections in 28 attempts.

Calibration worsens several frozen OLD-detector controls: their four-fixture
default mean increases from 0.106153° to 0.193571°. This is a retained finding,
not omitted from the benchmark. Full mechanism, controls, tests and per-scene
results: [`round2.txt`](../evidence/round2.txt).

Round 3 traces the held-out tail to **H09 seeds 0 and 1** (0.113444° and
0.101511° vertical), correcting Round 2's prose that named seed 4. The horizon
used TLS VPs fitted before final support reassignment. The solver now locks
family identities, then robustly refits selected VPs on their final supports
using length-weighted Cauchy angular IRLS. The horizon term remains enabled.
Held-out seeds 0–4 now have mean 0.006216° and worst 0.018802°; all 700 attempts
on seeds 0–49 pass with the same worst. Wrong-prior `01_input` still recovers
1426 px, with about 0.025° / 0.031° world error. Tests assert every held-out
seed, all 200 `01_input` focal/seed combinations, and gate rejection of the
original wrong H09 candidates. The held-out CLI prints every seed.

`python3.12 benchmark_quality.py --output ../evidence/round3_final.json` audits
all 50 seeds of 14 held-out scenes, four fixtures and `01_input` at four focal
priors. It reports gate-vs-independent-error confusion separately from returned
identity accuracy. Full before/after per-seed tables, gate-only discrimination,
frozen-support rejections, fixture corrections and limitations are in
[`round3.txt`](../evidence/round3.txt).

## Licence and patents

MIT covers the copyright in this code. **It does not grant patent rights**, and the
behaviour is covered. US 9,008,460 / 9,582,855 / 9,098,885 / 9,519,954 / 9,727,787
claim the pipeline and, in `9,582,855` claim 1, a purely automatic mode. The weight
values are unclaimed. Nothing here is free of patent risk before 2033-04-26, and
there is no clean position on `9,582,855` c1 unless Guided is the only mode shipped.
See `../evidence/patent_map.txt`. Engineering reading, not legal advice.

## Limitations

* On real photographs the port is not yet a usable corrector. It abstains on ~90%
  of frames (identity) and, ungated, loses to plain identity in 59% of paired
  comparisons. Its only advantage is `Auto`; `Full` loses even to naive
  rectification.
* `Auto` collapses above ~25 deg pitch and yaw. `Full` overshoots on scenes with a
  floor at 40 deg or more under a wide lens, because the floor's depth lines spawn a
  spurious lateral vanishing point that wins the "largest lateral" ranking.
* Family selection, not detection, is the dominant error: oracle family selection
  cuts `Full` median 3.66% -> 0.93% on real files, and round 12 closed none of it.
* `solve_ex` can return `ok=True` while 1-3 frame corners cross the projective
  horizon. The test exists in `frame_output` but is only applied when
  `framing=True`. 34 of 82 `ok=True` fuzz results were silently unusable.
* The energy remains a clean-room objective, not a recovered Adobe evaluator.
  The robust loss is our choice. Real fork detection, the Perkins barrier and
  the image-distortion term are not implemented.
* Length normalization in the energy is our choice; its denominator includes
  unused source segments. Only the picture-frame term uses pixel angles; the
  horizon term still uses normalized-coordinate angles.
* The detector is not verified pixel-identical; see the evidence corrections.
  The fallback only runs on empty results, so a detected long line can still
  prevent recovery of a separate short line lost to smoothing.
* Only the lateral horizontal family is aligned to the axis. Depth lines
  contribute through the horizon term.
* Source calibration assumes a centered square-pixel camera and perpendicular
  vertical/lateral families. Near-frontal geometry may not resolve focal; the
  prior is retained then. Biased or misclassified supports can bias calibration.
  Round 10 accepts only 80/900 attempts on Adobe-corrected real photographs,
  including two above 5% clone error; the synthetic results do not establish
  useful real-photo accuracy.
* No lens-distortion model or rendered crop/output-size selection; the caller owns
  the output canvas and any crop. Guided trusts the four supplied segments completely.
