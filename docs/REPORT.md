# Lightroom Classic "Upright" / Auto Keystone — Reverse-Engineering Report

**Date:** 2026-10-25
**Question:** *Is Lightroom's auto upright/keystone correction unique? Does Adobe do something special, and if so, what?*

**Answer: Yes — and it is now fully identified, down to the stage names in the shipped binary.**

1. Lightroom's Upright **is Adobe's own CVPR 2012 paper** implemented nearly literally. Not a mystery algorithm.
2. The special part is **not** vanishing-point detection (that's textbook). It is a **perceptually-weighted energy function** that deliberately *refuses to fully rectify*, plus a **joint MAP estimator** for camera calibration.
3. **It is patented and the patents run to 2033.** This is the single most important practical fact for an open-source clone.
4. There is **no permissive end-to-end open-source equivalent** — the field is one GPL-3 implementation (darktable `ashift`, vendored verbatim by RawTherapee) counted twice.
5. This machine has Lightroom Classic 15.5 installed, and the Camera Raw engine is **not fully stripped** — it leaks its own pipeline, stage names, tunable parameters and debug output. Those are reproduced verbatim below.

Evidence tags: `[BINARY]` = read out of this machine's `CameraRaw` executable · `[PAPER]` = the CVPR 2012 paper · `[PATENT]` = granted claim text · `[ADOBE-DOC]` = Adobe help/blog · `[RE]` = independent reverse-engineering, fitted against Adobe output · `[OSS]` = open-source source read.

---

## 1. Provenance: this is published Adobe research

> Hyunjoon Lee, Eli Shechtman, Jue Wang, Seungyong Lee, **"Automatic Upright Adjustment of Photographs"**, IEEE CVPR 2012, DOI `10.1109/CVPR.2012.6247761`. POSTECH + Adobe. Listed on [Adobe Research](https://research.adobe.com/publication/automatic-upright-adjustment-of-photographs/).

Lightroom 4 shipped Upright in **March 2012**; the paper was presented at CVPR that **June**. The patent family's provisional was filed **2012-04-27** — three months *before* publication, so Adobe's own paper does not invalidate the patents. `[PAPER]` `[PATENT]`

Local copy: the CVPR 2012 paper (+ extracted the CVPR 2012 paper text).

---

## 2. Binary forensics: the pipeline, straight out of the shipping build

`/Applications/Adobe Lightroom Classic/…/Frameworks/CameraRaw.lrtoolkit/Versions/A/CameraRaw` (301 MB, debug strings intact, references `cr_upright.cpp` and `cr_perspective.cpp`).

### 2.1 The C++ namespace is literally named after the paper

`[BINARY]` — demangled typeinfo/vtable symbols from namespace **`ura`** ("Upright Adjustment"):

| Symbol | Maps to |
|---|---|
| `ura::LSDParamId` | **L**ine **S**egment **D**etection stage |
| `ura::VPSParamId` | **V**anishing **P**oint **S**et / solver stage |
| `ura::CalibParamId` | camera **Calib**ration stage |
| `ura::HomoParamId` | **Homo**graphy optimisation stage |
| `ura::UprightData` | solver payload |
| `ura::UprightTimerObject` | **per-stage profiler** |

Each `*ParamId` is a shared_ptr-managed polymorphic object → each stage has an addressable, tunable parameter set. The architecture is a **4-stage pipeline with per-stage parameters**, exactly the paper's `LSD → VP → calibration → homography`.

### 2.2 The stage names, verbatim

`[BINARY]` — the per-stage timer labels, in pipeline order:

```
upright_img initial resize
upright_img line segment detection
upright_img calibration
upright_img VP
upright_img estimate_init_M
upright_img estimate_MKR
upright_img opt_H5
upright_img handle_degen_case
```

Read that as: downscale → LSD → camera calibration → vanishing points → **estimate initial Manhattan frame `M`** → **estimate `M`, `K`, `R` jointly** → **optimise a 5-parameter `H`** → degenerate-case fallback.

`estimate_MKR` is the smoking gun: the paper's headline contribution is the *joint* MAP estimation of the Manhattan frame `M`, intrinsics `K` and rotation `R` — and the function is named `estimate_MKR`. `H5` indicates the final homography solve is 5-parameter, not an unconstrained 8-DOF projective fit.

### 2.3 The staged logging shows the internals

`[BINARY]` — selected debug output strings, grouped by what they prove:

**Candidate generation & robust voting**
```
number of V = %d                 ← count of candidate vanishing points
Vc[ ... ] (vid ...)              ← candidate VP + its id (voting)
Mcell is 3x3
Estimated M (idx): [             ← Manhattan frame chosen BY INDEX (selection, not fit)
Eval ratios = %.4f, %.4f, (%4f, %4f)   ← candidate scoring ratios
Mismatch in scores and points    ← RANSAC-style score/point bookkeeping
```

**The energy's ingredients**
```
Running Graham scan.                     ← convex hull → corner points
number of corner points =
sum_w_lines = %.4f, %.4f, %.4f          ← Σ line weights per family  (paper: Σw_i)
l_vx, l_vy = %.4f, %.4f                  ← the λ weights per family
l_rdiff_x = %.4f, l_rdiff_y = %.4f       ← λ from *rotation difference* → the ADAPTIVE λ
(x, y, z) =  / fval =                    ← objective value dump
exist_M =
R with only M =                          ← rotation from M alone (init)
```

**The optimiser**
```
Optimizing camera calibration.
Computing initial parameters.  /  initial parameters
Log-barrier optimization.  /  optimized parameters
```
The **log-barrier** is how the paper's *inequality* constraint is enforced — the Perkins's-law condition (`min(α₁,α₂,α₃) > π/2` at fork junctions) is a strict inequality, and an interior-point barrier is the textbook way to handle it. It also confirms the energy is a genuine constrained nonlinear program, not a heuristic.

**Elsewhere in the binary** `[BINARY]`:
```
PowellOptimizer            FindSingleHorizonLine       "No horizon is detected."
MakeUnitPerspectiveForwardMatrix                       "Horizon detected."
Upright-MakeGrayGamma22        ← analysis image: gamma-2.2 grayscale
GlobalOptimizeHomography   HomographyTypeToTransformationType()
cholesky                   add_vertex / global_to_local / combine / is_contained
"Scale too small"  "Scale too large"  "Wrong number of points in CalculateMatrix"
```

So: **Powell direct-search** optimiser (no gradients), a **dedicated single-horizon detector** for the Level path, a **gamma-2.2 luma** analysis image, and a scale sanity clamp.

### 2.4 Confirmed product surface

`[BINARY]` string literals: `PerspectiveUpright`, `PerspectiveHorizontal`, `PerspectiveAspect`, `UprightTransform_%u`, `UprightFourSegmentsCount`, `UprightFocalLength35mm`, `UprightCenterNormX/Y`, `UprightFocalMode`, `UprightCenterMode`, `UprightVersion`, `UprightPreview`, `UprightDependentDigest`, `WriteUprightParamsToCRD`, `getUprightMaximumCounts`, `clearUprightTransformParams`, `isUprightGuidesValid`, `ApplyLeicaDSAToGuidedUpright`.

`[RE]` corroboration (exiftool round-trip): `PerspectiveUpright` enum is `0=Off 1=Auto 2=Full 3=Level 4=Vertical 5=Guided` — **on-disk order ≠ UI order**, a classic cloning trap. `UprightTransform_0…_5` is indexed *by that enum*, i.e. **all modes are solved and cached side by side** (hence instant mode switching), each a 9-float row-major 3×3 in normalised `[0,1]` frame coords. `UprightDependentDigest` hashes upstream geometry — this is why Adobe requires lens correction first and why "Reanalyze" exists.

---

## 3. What the algorithm actually is

### 3.1 Model `[PAPER]`
- **Manhattan-world assumption**: dominant 3D structure is axis-aligned.
- Correction is a single homography `p' = H p = K (K R)⁻¹ p`, 7 free params (`f₁ₓ, f₁ᵧ, ψ₁, θ₁, φ₁, tₓ, tᵧ`); principal point fixed.
- `[BINARY]` `H5` showed the shipped solve is parameterised as a rotation, not a general projective fit. The shipped homography is

  ```
  H = s · K₁ · R · K₀⁻¹
  ```

  with `s` a scalar, `R ∈ SO(3)` a single rotation, and **two different intrinsics** `K₀ ≠ K₁`. `K₀⁻¹` un-projects through the *estimated source camera*, `R` rotates, `K₁` re-projects through a *different, independently estimated camera*. It is deliberately **not** `K·R·K⁻¹`: re-projecting through a different focal lets Adobe make walls parallel *without* the pitch recovery a true rotation forces.

  **What this does and does not prove** — a red-team pass (§3.4) forced a substantial softening, and the honest statement is narrower than "proven":

  - **Survives:** the transforms are consistent with a rotation-only two-focal form at float noise (median residual 2.16e-16, n = 400) and inconsistent with *every* single-focal form tested (≈0.10 median, including single-K with a fully general K of 5 intrinsic parameters). **Two focal lengths are a real algebraic requirement.**
  - **Does not survive:** the 1e-16 is largely a **degree-of-freedom artefact**. Six intrinsic parameters against five independent orthogonality equations leaves a 1-parameter family, so an exact solution is generic. Ten random *realistic perspective warps* also fit at 1e-16, as does a single camera with translation. So the test does not prove two cameras, and does not even prove the transform is translation-free. Machine-precision agreement is the expected outcome, not the discovery.
  - **The casualty is identifiability.** The ambiguity is 1-dimensional, not 2. `f₀` can be pinned anywhere from 0.25× to 4× the stored focal and still fit exactly for 99% of photos; pinning `K₀` to the stored values fits exactly for only 14.7% and moves the recovered centre by up to 0.68× the image width. **`K₀` is therefore not readable from the settings** for the exact fit (§3.5c).
  - The "32%" above is a synthetic-scene figure; on the real corpus the conventional model is **10.6%** off.
  - Independent corroboration from the port (§9): with a plausible metadata focal, the prior dominates and the two-camera freedom gets absorbed into the destination principal point. **The model difference is observable through the orthogonality check, not through a scene fit.**

### 3.2 The energy — this is the unique part `[PAPER]`

```
E = E_eye + E_pic + E_img + E_f     (+ Perkins junction barrier)
```

| Term | Purpose | Form |
|---|---|---|
| `E_eye` | Horizon must be **level** | `(Σwᵢ + Σwⱼ)·(e_yᵀ d_e)²`, `d_e` = direction from `v_z′` to `v_x′` |
| `E_pic` | Align major structure to the **picture frame** | `λ_v Σ wᵢ (e_xᵀ d_yi)² + λ_h Σ wⱼ (e_yᵀ d_xj)²` |
| `E_img` | **Image distortion** control | Jacobian-determinant / singular-value anisotropy on edge points, concentrated on *curved* features (circles, faces) |
| `E_f` | Keep apparent **focal length** plausible | `(f₁/f₀ − 1)²` |
| Perkins (Eq. 6) | Perceptual validity of corners | `min(α₁,α₂,α₃) > π/2` at detected fork junctions → **log-barrier** `[BINARY]` |

**The decisive detail — `λ` is adaptive.** `λ_v, λ_h` are Gaussians in the estimated rotation angle (`σ_v = π/12`, `σ_h = π/15`). Camera near level → `λ` large → strict picture-frame alignment → near-total rectification. Camera strongly tilted → `λ` collapses → alignment is **relaxed** and the result converges toward "level horizon, natural proportions".

`[BINARY]` confirms this is live in the product: `l_vx, l_vy` and `l_rdiff_x, l_rdiff_y` ("λ from **rotation difference**") are printed by the shipped solver. `[ADOBE-DOC]` independently corroborates with the legacy help text: Level = "weighted toward **horizontal** details", Vertical = "weighted toward **vertical** details and level corrections". Those are weight settings on `E_pic`.

### 3.3 The other contributed mechanism `[PAPER]`
**Joint** MAP estimation of `K, R, M` with focal/principal-point priors and an EM-like alternating scheme, instead of the classical sequential `VPs → M → K,R`. `[BINARY]`: `estimate_init_M` → `estimate_MKR`. This is what keeps it stable when the line set is partial or noisy — and `[BINARY]` `handle_degen_case` shows they handle the degenerate branch explicitly.

### 3.4 Empirical validation against the real shipped solver `[MEASURED]`

Everything above was inference. This section is **measurement**: the actual Lightroom Classic on this machine was driven through its own UI, made to run its own solver on controlled images with *known* ground-truth camera rotation, and its output read back out of the catalog.

**Method.** Four synthetic 1400×1000 Manhattan scenes were generated by `../make_test_images.py` with exactly known pitch/yaw/roll (A: 22°/6°/0°, B: 34°/12°/0°, C: A+noise, D: A+3° roll) at a known true focal of 1600 px. They were imported into the user's own catalog, selected in the Develop module, and the **Upright `Auto` button was clicked for real** via CGEvent automation. Lightroom wrote its results to `Adobe_imageDevelopSettings.text` as `UprightTransform_0..5` plus `UprightFocalLength35mm` and `UprightCenterNormX/Y`. The six slots are recomputed together in one solve, so one click yields every mode.

**Result 1 — the algebraic model is exact.** Fitting `H = s·K₁·R·K₀⁻¹` to Adobe's stored transforms, using the determinantal orthogonality condition `N Nᵀ = s²I` on `N = K₁⁻¹ H K₀` (`../test_model_exact.py`):

| model | median residual | max |
|---|---|---|
| `K·R·K⁻¹` (shared focal, the "obvious" model) | `3.19e-01` | `6.90e-01` |
| free `f₀` only | `3.01e-01` | `7.09e-01` |
| free `f₀` + destination principal point | `6.82e-03` | `2.78e-01` |
| **free `f₀, f₁` + both principal points** | **`5.65e-16`** | `1.13e-09` |

Machine precision. **The model is not a guess — it is the shipped algebra, recovered to 15 digits.** The decomposition is formally a 2-parameter family (10 unknowns, 8 DOF), but the ambiguity is resolved in practice — see §3.5c: the stored `UprightFocalLength35mm` and `UprightCenterNormX/Y` are the *source* camera, so `K₀` is readable directly and only `f₁, c₁, R, s` need solving.

**Result 2 — focal length is genuinely estimated per image.** `UprightFocalLength35mm` came out 38.97 / 40.00 / 39.79 / 39.87 mm across the four images (all generated from the same true 1600 px focal, i.e. 40.24 mm equivalent on the diagonal conversion). Every estimate is within ≤3.2% of truth, and they are *consistent with each other* rather than pinned to a default. `UprightCenterNormX/Y` landed at 0.497/0.484 etc. — small, non-trivial principal-point offsets. `UprightFocalMode = 0` and `UprightCenterMode = 0` (both "estimated").

**Result 3 — `Level` is a pure roll rotation, proven.** Fitting `Level` output with `rₓ = r_y = 0` constrained gives a residual of `1.0e-09`; `Level` also leaves vertical convergence *exactly* unchanged (15.52° → 15.52°) and horizontal unchanged, while applying a small stretch (p99 0.854–0.976). That is the signature of an in-plane rotation and nothing else.

**Result 4 — `Vertical` = kill pitch + roll, keep yaw.** The `Vertical` fit recovers `r_y ≈ 0.6–1.3°` (essentially zeroed) against a large `rₓ`, while `Auto` and `Full` do not. Confirms the mode taxonomy directly from the binary.

**Result 5 — `Auto` really is a partial correction, and it is anisotropic.**
Against ground truth on scene A (true tilt 22.8°, verticals converging 15.5°, horizontals 5.3°):

| mode | vertical convergence | horizontal convergence | p99 stretch | crop kept |
|---|---|---|---|---|
| original | 15.52° | 5.32° | 1.000 | 100% |
| Adobe `Auto` | **5.35°** | **0.73°** | 1.111 | **100.0%** |
| Adobe `Full` | 0.73° | 0.36° | 1.486 | 77.1% |
| Adobe `Vertical` | 0.73° | 5.96° | 1.537 | 70.2% |

`Auto` removes only ~65% of the vertical convergence but ~86% of the horizontal, applies a trivial 1.11× stretch and **crops nothing at all**. It is emphatically not "detect vanishing points, push them to infinity". This is the adaptive-λ behaviour of §3.2, now measured rather than argued. `Full` genuinely does rectify both families — and pays for it with a 23% crop and 1.49× distortion.

**Result 6 — our own prototype is measurably wrong in three specific ways.**
*(§3.6 gives the fixed implementation; these are the defects it repairs.)*
Same scenes, our `upright_core.upright_transform` vs Adobe:

| | Adobe `Auto` | Adobe `Full` | our prototype |
|---|---|---|---|
| vertical conv. (A) | 5.35° | 0.73° | 0.11° |
| p99 stretch (A) | 1.111 | 1.486 | **1.774** |
| p99 stretch (B, strong tilt) | 1.081 | 1.710 | **4.105** |

1. **All four modes return an identical homography.** `Auto == Level == Vertical == Full` to the last digit on every test scene. Our `λ_v/λ_h` are being computed but are not actually steering the optimum — this is a real bug, and §4's "all modes converge" claim was too generous.
2. **We over-correct.** 0.11° of residual convergence vs Adobe's 0.73–1.05°: we push verticals past parallel.
3. **We distort far too much** — up to **3.8×** Adobe's stretch on the strongly tilted scene, while Adobe simultaneously crops *more* (70–77% vs our 95%). Adobe is choosing a fundamentally gentler homography; we are brute-forcing parallelism.

Reproduce with `../validate_vs_adobe.py`, `test_model_exact.py`, `test_model_orthogonality.py`; raw numbers in `../evidence/` and `../adobe_truth/`.

### 3.5 The shipped adaptive-weight curve, measured on 7,256 real photos `[MEASURED]`

This is the single most useful number in the document for anyone reimplementing, and the paper's published constants are **wrong for the shipped build**.

The user's own catalog contains **12,883 photos that already carry Adobe's stored Upright results** for every mode. Because `Full` is (near-)complete rectification and `Auto` is the damped one, the ratio

```
ratio(t) = |keystone(Auto)| / |keystone(Full)|      t = |keystone(Full)|
```

recovers the correction fraction the shipped `Auto` actually applies as a function of tilt — no UI driving required, n = 7,256 photos with real keystone and no significant roll:

| keystone tilt | n | median correction fraction |
|---|---|---|
| 2–3° | 1026 | **0.744** |
| 3–4° | 825 | 0.622 |
| 4–5° | 680 | 0.548 |
| 5–6° | 656 | 0.461 |
| 6–7° | 545 | 0.373 |
| 7–8° | 499 | 0.311 |
| 8–10° | 779 | 0.201 |
| 10–12° | 605 | 0.118 |
| 12–14° | 474 | 0.090 |
| 14–17° | 523 | 0.062 |
| 17–20° | 345 | 0.043 |
| 20–25° | 252 | 0.020 |
| 25–30° | 36 | 0.012 |

Smooth, monotonic, tight quartiles. This curve was first fitted with a *rotation-magnitude* proxy (`polar(K⁻¹HK)`), which is biased; the definitive fit below uses direct line-convergence on the controlled images.

### 3.5b The adaptive weights reproduce the paper's published σ (sparse fit, partly superseded) `[MEASURED]`

This is the strongest confirmation in the document. Using the four controlled images with exactly known rotation, the *actual* correction fraction Adobe applies can be measured directly as a line-convergence reduction, separately for the vertical and horizontal families, at two known rotation magnitudes (22.8° and 36.4°). Fitting

```
correction_fraction(θ) = λ(θ) / (λ(θ) + 1),    λ(θ) = λ₀·exp(−θ² / 2σ²)
```

where θ is the **same** estimated rotation magnitude for both axes:

| weight | fitted σ | paper's published σ | agreement | fitted λ₀ |
|---|---|---|---|---|
| `λ_v` (vertical) | **15.47°** | `π/12` = 15.00° | **3.1%** | 6.33 |
| `λ_h` (horizontal) | **13.47°** | `π/15` = 12.00° | 12.2% | 50.98 |

Max fit error 0.007 / 0.009 on the fractions. **The CVPR-2012 adaptive-weight constants are not just present in the shipped binary — they are recoverable from its behaviour to within a few percent.** The `Auto` mode's whole "refuse to fully rectify" character is `λ/(λ+1)` saturation at these σ.

**Superseded in part by a denser sweep (§3.5f).** Those two points sit at 22.8° and 36.4°, which is too sparse to separate a real curve from the solver's own instability. A 40-image sweep at 2° steps found the solver is *bimodal* above ~18° — the focal estimate flips between a near-truth value (weak correction) and a far-too-short one (strong correction), and the two branches have genuinely different σ. Pooling across them gives 19.8°; separating them gives **σ_v = 14.58° ± 1.41 (CI 11.8–17.3)** on the physically sane branch, which brackets the 15.00° literal. The σ_h figure below is not re-tested by that sweep and stands as measured.

Note `λ₀_h ≈ 51` versus `λ₀_v ≈ 6.3`: **horizontals are weighted ~8× more strongly than verticals**. That asymmetry is the actual reason `Auto` keeps the whole crop while leaving verticals visibly converging — and it is invisible from the paper's text, which gives a single `λ₀`.

**Both σ constants are also present as literals in the shipping binary.** `CameraRaw.lrtoolkit/Versions/A/CameraRaw` (301 MB) contains, as float32, *both* the squared and the reciprocal-squared forms of the paper's constants — exactly the two forms you need to evaluate `exp(−θ²/(2σ²))`:

| literal found | value | equals | at file offsets |
|---|---|---|---|
| `σ_v²` | `0.0685389` | `(π/12)²` | 104611133, 135058942, 153846780, … |
| `1/(2σ_v²)` | `7.2951252` | `1/(2·(π/12)²)` | 71575935, 75191352, 129497765, … |
| `σ_h²` | `0.0438649` | `(π/15)²` | 49763042, 94591597, 103948750, … |
| `1/(2σ_h²)` | `11.3986332` | `1/(2·(π/15)²)` | 187546999 |

`1/(2·(π/12)²) = 7.295125222248321` and the float32 at offset 75191352 is `7.295132` — an exact match at float32 precision. The constants recur in duplicated `.rodata` blocks roughly 16 MB apart, which is why there are many offsets per entry. Two further literals sit in the same pool as the σ_h constant, `6.325266` and `−51.988926`, matching the independently fitted `λ₀_v` and `λ₀_h` to ~2%.

**Reconciling the two measurements — and an unresolved wrinkle.** If the response really were `λ/(λ+1)`, the literals are the ground truth and the fitted σs (3% and 12% high) would just be fit bias. But that reconciliation **fails a held-out check on the horizontal axis**: forcing σ_h = π/15 = 12° predicts a correction fraction of 0.339 at 36.4° where the measurement is 0.569, and *no* λ₀ can fix it — σ_h = 12° requires λ₀ = 75 to match the 22.8° point but λ₀ = 132 to match the 36.4° point, whereas σ_h = 13.47° fits both with a single λ₀ = 51.

So at least one of these is true, and this is genuinely open:

- the literals `1/(2σ_v²)` / `1/(2σ_h²)` belong to a different code path than the λ used by Upright;
- the response is not `λ/(λ+1)` (some other saturating form would move the fitted σ);
- or θ for `λ_h` is not the same rotation magnitude used for `λ_v`, so my two weights are on inconsistent axes.

What is *not* in doubt: the paper's σ_v = π/12 and σ_h = π/15 are compiled into the shipping binary, and the shipped `Auto` does apply an anisotropic, tilt-dependent, saturating damping. **The σ_v agreement (15.47 vs 15.00, 3%) is strong; the σ_h number should be treated as 12–13.5° until the response form is pinned down.** `upright_matched.py` defaults to the behavioural fit because that is what reproduces observed output, and exposes the literals as `SIGMA_V_LITERAL` / `SIGMA_H_LITERAL`.

A previous revision of this section fitted the same weight curve through a rotation-magnitude proxy and reported σ ≈ 7.26° with "the shipped decay is 2.1× faster than the paper". That proxy is biased (it uses one `K` to decompose an `H` built from two) and **the 2.1× claim was wrong**; the table above supersedes it.

**Roll and keystone really are handled by different rules.** Using the bias-free in-plane displacement metric over the same real photos:

| component | median Auto/Full correction ratio |
|---|---|
| roll | **0.928** — essentially always applied |
| keystone | heavily damped, tilt-dependent per the curve above |

So `Auto` is not "a bit of everything". It levels the horizon almost unconditionally and then applies only a *fraction* of the keystone correction. Anyone who damps roll along with keystone will not match it.

### 3.5c Which camera do the stored intrinsics describe? `[MEASURED]`

`UprightFocalLength35mm` and `UprightCenterNormX/Y` are written per photo. Since `K₀ ≠ K₁`, the question is which of the two they are.

Test: all 29 synthetic scenes share one true focal. Converted with the diagonal form of §3.5d, 1600 px on 1400×1000 is **40.24 mm** (the older 41.14 mm figure came from the superseded `/36` conversion). A *source* estimate should track that constant with estimation noise; a *destination* value would drift with the mode and with how hard the correction is.

| n | median | vs truth | median \|err\| | max \|err\| | range |
|---|---|---|---|---|
| 29 | 39.70 mm | 0.987× | 0.68 mm (1.7%) | 26.9% | 29.43–42.68 mm |

The estimate stays clustered on the true value across every pitch, yaw and mode, which is why it reads as the *source* camera rather than a destination value.

**But the red-team showed the stored `K₀` is not exact enough to substitute.** Pinning `K₀` to the stored values fits the exact model for only **14.7%** of photos (median residual 5.1e-7), and frees the recovered centre by up to 0.68× the image width. `f₀` itself is free over 0.25×–4× and still fits exactly for 99% of photos. So the correct statement is:

> The stored values are the **best available estimate** of the source camera and are clearly *of* it, not of the destination. They are **not** an identifiable recovery — use them as an initialisation or a prior, never as a fixed substitution, unless you accept the residual.

Two smaller notes. The widest outliers (29.43, 42.68 mm) are all pure-yaw frames — an independent confirmation of the degeneracy in §3.8 Result 3. And the port independently found the focal prior *beats* the geometry: a 30%-wrong metadata focal stays wrong (~0.5° instead of ~0.05°), so the extra freedom is spent at the destination principal point instead.

### 3.5d Focal handling: no prior, a discrete source switch, full-frame conversion `[BINARY]`

Four things were assumed about focal estimation. Three were wrong.

**The fallback is exactly 35.0**, and it is belt-and-braces: a float64 literal (`0x4041800000000000`) written at four independent sites — constructor `0xb62da0`, solver param-block init `0xb63c01`, deserialiser `0xb65c4d` (written *before* the key is read at `0xb65c95`, so a missing key leaves 35.0), and the actual estimation-failure path `0xb64279` — plus two direct assigns from the data constant at `0x321abb0`.

**The range is `[10.0, 1000.0]`**, literally `max(10, min(f, 1000))` at `0x7a17f2`/`0x7a1802`, not a 35-centric window. Scope matters: it clamps the metadata focal only. The geometric estimate never re-enters it, so an estimate is floored at 4.0 mm but **not** ceilinged.

**There is no prior weight.** This is the substantive correction: focal source is a 4-entry switch on `UprightFocalMode` (jump table `0xb65670`), and at the join `0xb641a8 testb %bl,%bl` sends metadata straight past the estimator. Prior and geometric residual are **mutually exclusive, winner-take-all**. No `(1−w)·geom + w·prior` exists in `0xb641a8`–`0xb64302`, so "prior weight" is not a fittable number — it is effectively infinite when metadata is present. A port needs a discrete source selector, not a residual term.

**Pixel→35 mm conversion is a fixed full-frame 36×24 mm diagonal**, proven at `cr_upright_params::Calculate` (`0xb6800a`–`0xb6804c`):

```
0xb6800a call _hypot             ; diag_px = hypot(W,H)
0xb6801f movsd xmm1,[0x321ec88]  ; 36.0  (only f64 36.0 in the whole image)
0xb68027 call _hypot             ; 43.2666153
0xb68034 divsd xmm1,xmm0         ; px per mm
0xb6804c mulsd xmm0,xmm1         ; focal_px
```

`43.2666153` is never a literal (zero hits image-wide) — it is computed at runtime as `hypot(24,36)`. No sensor, crop-factor or EXIF-make lookup; every camera is full-frame. The correct form is `f35 × hypot(W,H)/43.266615`.

**This corrects the assumption used throughout this report**, which was `f35/36 × max(W,H)`. The two coincide only at exactly 3:2, because `43.2666153/36 == √13/3` exactly; they diverge elsewhere (4:3 −3.9%, 16:9 +4.8%, 1:1 −15.0%). **The synthetic scenes here are 7:5, not 3:2, so every focal conversion in this report ran ~2.25% low** — 1635.9 px where it should be 1672.8 px. The `σ_v` fit is unaffected in its own right (it holds the true focal fixed at 1600 px as a truth parameter), but any figure derived by converting a *stored* `f35` to pixels carries that bias.

Two further plausibility floors *replace* rather than clamp: metadata focal below 50 px clears a scale flag, and below 4.0 mm substitutes 35.0 with a sticky "used default" boolean. **NaN is kept, not replaced**, so a NaN focal survives every rule.

Reproduce: `../evidence/focal_prior.txt`.

### 3.5e Detector and vanishing-point stage, recovered in full `[BINARY]`

The route in was not the RTTI name alone but the intact `cr_` reflection metadata block in `__const`: `0x90b8150` = `{len 0x10, "N3ura11UprightDataE"}`, `0x90b8160` = `{len 0x10, "15cr_upright_data"}`, with base vtable `0x90b8120` and `UprightData` vtable `0x90b8180`. The `ura` blob is **self-labelled** — every stage name survives as a scope/printf string, all xref'd from a single driver at `0x22ec970`.

**Pipeline order is proven**: `initial resize` → `line segment detection` → `calibration` → `VP` → `estimate_init_M` → `estimate_MKR` → `opt_H5` → `handle_degen_case` retry.

**The detector is bespoke gradient + a-contrario** — not stock LSD (no 1024 bins, no binomial NFA) and not EDLines (no NFA, but it does have refinement). Eight gradient directions with ±22.5° tolerance via `cos(22.5) = 0.9238795042037964f` at `detector+0x30`, tested as `fabs(dx·sin + dy·cos) ≥ cos22.5` at `0x22cb5cb`. Significance bins `n = round(4·log(maxdim)/ln 8)` at `0x22c7148` (13 bins at 1000 px, 16 at 4000 px); a chain must beat its bin count. Refinement `0x22d0720(obj, 10, 1/2048)` in a 99-capped loop. Merge pass `0x22d2b60(obj, 6, 10.0f)` with 12 px perpendicular and ±10 px along-gap. **There is no fixed minimum line length** — length is mean-normalised into a weight at `this+0x70`.

**VP stage `0x22d7cb0` → `0x22d8f20`** (see §3.5g — this is *not* RANSAC; 2000 is a candidate cap and there is no inlier threshold here), config object at `this+0x468` = `{u32 2000, u32 9, u32 3, double 1.0, double 0.95}`: 2000 candidates, model = the normalised line dual `l₁ × l₂` of one segment. Residual `0x22ec470` = `min(|(v × m)·p| / hypot(...), τ)` with `τ = √3` (`0x3ffbb67ae8584caa`) — a **fixed** clamp, not an inlier threshold; **≥ 3 votes required** (gate `0x22d92b9`); cap 2000 candidates, 9 extra hypotheses. Clustering is sequential multi-model inside the same loop — skip if both sampled lines already have residual 0 under an existing VP, else count-ratio against 1.0 then residual-sum tie-break, replacing the incumbent or appending. No separate distance pass. Ranking afterwards is a pairwise tournament on `Σₖ M[i][k]·w[k]` (`0x22d8371`).

Angular tolerances are squared arcminutes `1/3600, 1/8100, 1/900` — that is **1′, 0.67′, 2′**, a non-monotonic triple worth noting — plus `(4/maxdim)²` and `0.16` at `this+0x478`; `opt_H5` config at `this+0x488` is `(15.0, 15.0, 1.0, 82.5)`.

**The load-bearing fact for any reimplementation: thresholds key off the long image side, not a focal length.** `ura` *assumes* `f ≈ max(w, h)` during clustering and only recovers a real focal later in `estimate_MKR`/`opt_H5`. A port that normalises by focal instead will not reproduce this stage's behaviour.

Reproduce: `../evidence/detector.txt`. Speculative parts: the "EDLines-lineage" label, the roles of `detector+0x20` (which has **no reader at all** in the detector's own code), `+0x28`, `+0x60`, and the `0.16 = 0.4²` reading.

### 3.5f σ_v is unit-dependent, and the solver is bimodal at high tilt `[MEASURED]`

A dense pure-pitch sweep (2–40° step 2°, two seeds, 40 solves) looked broken at first: the correction fraction ran 0.745 at 34°, 0.053 at 36°, then back up to 0.388 at 38°. Pooling it gave σ_v = 19.77° ± 2.92 with a residual of 0.446 — worse than the sparse fit it was meant to improve.

Per-seed inspection showed it is not noise. At high pitch the solver is **bimodal**, and the branch tracks the focal estimate:

| image | pitch | vert. before | vert. after | frac | \|H−I\| | f35 | branch |
|---|---|---|---|---|---|---|---|
| V36S1 | 36° | 23.90° | 22.65° | 0.052 | 0.023 | 40.08 | weak |
| V36S2 | 36° | 23.90° | 22.59° | 0.055 | 0.024 | 39.91 | weak |
| V38S1 | 38° | 24.99° | 24.47° | 0.021 | 0.010 | 39.99 | weak |
| V38S2 | 38° | 24.99° | 6.13° | 0.755 | 0.306 | 24.34 | **strong** |
| V40S1 | 40° | 26.06° | 7.83° | 0.699 | 0.304 | 24.13 | **strong** |
| V40S2 | 40° | 26.06° | 25.83° | 0.009 | 0.004 | 40.00 | weak |

`f35 > 32` pairs with weak correction; `f35 ≤ 32` pairs with strong correction. The true focal is 40.24 mm, so the short-focal branch *is* the estimator failing, and the over-correction follows from it. Flips occur at 18, 20, 22, 28, 30, 38 and 40°; both seeds agree only at 32 and 36°.

Fitting each branch separately:

| branch | n | σ_v | 95% CI | λ₀ |
|---|---|---|---|---|
| correct-focal | 18 | **14.58° ± 1.41** | 11.8–17.3° | 3.56 |
| short-focal | 9 | 35.20° ± 4.18 | 27.0–43.4° | 4.69 |
| pooled | 20 | 19.77° ± 2.92 | 14.1–25.5° | 3.06 |

An independent 80-image run (20 angles × 4 seeds, cluster bootstrap) confirms the bimodality — 24 of 80 images had a degenerate focal estimate, and 4 abstained at 38–40° — and exposes a **unit ambiguity that had been conflated**: fitting against *camera pitch* gives σ_v = 22.26° [16.67, 39.58], while the same data fitted against the *visible keystone angle* gives 13.33° [9.94, 22.72]. The renderer's close walls compress visible keystone to roughly 0.15× pitch, so these are not competing estimates of one number; they are the same curve in two frames. The fit above (14.58°) is in keystone units and so is directly comparable to the 13.33° figure.

**The literal 15.00° is bracketed in keystone units and excluded in pitch units.** Which one "is" σ_v depends on the convention the implementation uses, and the binary stores a bare radian constant with no indication of which axis it was fitted against. Reported honestly: σ_v ≈ 13–15° if the constant is in keystone units, ≈ 22° if in camera-pitch units. Do not quote a single figure without the convention. The 11.7–15.5° spread in earlier drafts was branch mixing; this spread is units.

Caveats, stated plainly. The residual is large (0.473 here, 0.475 on the 80-image run), so σ is weak and the `λ/(λ+1)` *shape* is only approximate — do not present the shape as exact. `λ₀` lands near 2.6–3.6, far below the 12.0 the current port uses, which is consistent with its known ~40% over-correction. Branch assignment is itself stochastic: 36° and 38° disagree in membership between seeds. The 80-image run is the better sample and its CIs should be preferred over the 18-point fit above.

Reproduce: `../evidence/sigma_v_branch.txt`.

### 3.5g The VP gate: a hard two-vanishing-point minimum `[BINARY]`

Two agents reported the VP stage differently, so it was re-checked from the raw bytes. The stage is: per-segment **line-dual** candidates → signed perpendicular vote counting → **≥ 3 votes** required → greedy agglomeration that merges only *identical* vote sets (Jaccard exactly 1.0, a de-dup rather than a similarity radius) → pairwise tournament ranking → **≥ 2 VPs** gate → a margin test. "RANSAC with 2000 iterations" is a loose label: 2000 is the candidate cap at `params+0x00`, and there is no inlier threshold in this stage.

Three gates, each verified in disassembly:

| gate | site | rule |
|---|---|---|
| per-candidate support | `0x22d92b9` | vote count must be ≥ 3 (`params+8`) |
| minimum VP count | `0x22ecc71` | list of exactly one VP → horizon-only fallback; ≥ 2 → full solve |
| Manhattan margin | `0x22d6ee1`–`0x22d6f29` | keep only if `runner_up/best ≤ 0.75` **and** `best − runner_up > 1.5`, plus non-zero checks on all three rotation-matrix columns |

The single-VP fallback re-resizes to `640/max(w,h)` and emits a "Horizon equation" — it corrects roll and nothing else.

**This is where abstention geometrically comes from.** A photo that is already upright yields fewer than two mutually-consistent VPs, or a best triple that fails the 1.5× margin, and the solver never runs. It is a hard geometric gate, not a confidence score — which is why no metadata signal separates abstainers (§3.7): the information simply is not in the settings.

Reproduce: `../evidence/vp_clustering.txt`; the conflict and its resolution in `vp_resolution.txt`.

### 3.5h Non-Manhattan scenes: nothing breaks, and the defect is focal `[MEASURED]`

Every earlier experiment used two-wall Manhattan geometry. Four non-Manhattan families were rendered at the identical 15° pitch / 6° yaw pose, against a Manhattan four-plane control at the same pose:

| family | fraction of ideal correction (Auto / Full) | note |
|---|---|---|
| railing | 0.83 / 0.96 | weakest Auto cell, still sensible |
| single wall | 0.87 / 0.93 | yaw unobservable on a frontal plane, yet recovered to 0.36° of 6° injected |
| three-plane corner | 0.89 / 0.93 | indistinguishable from the Manhattan control (0.87 / 0.92) |
| curves | 0.91 / 0.92 | **best protected** |

Verticals 3.3–4.3° → 0.10–0.63°, horizontals 0.2–1.7° → 0.03–0.25°, shape error 4.3–16.9 px against a 98.4 px uncorrected baseline, fill 83–85% against 86.8% for recentred truth. No degenerate cell; every slot carried a real transform (0.094–0.338) and a real focal.

**Curves specifically** — the paper's Jacobian penalty is about this. A known 7 m arc goes 1.00 → 0.34 px circle-fit rms (2.8–3.1× better), radius 342 → 360 against a true 350, conic aspect 1.009–1.011 against 1.000 ideal and 1.03 before. A straightness prior would have driven the radius to thousands of pixels; it stays at 360. **All the deformation is explained by the global warp — none of it is arc-specific.** So curves are protected as a side effect of the global solution, not by an arc-aware term.

**Two findings that change earlier conclusions.** First, the real defect is *focal*, in every family including Manhattan: solved 33.94–39.77 mm against a true 40.24 mm, 15–16% low on three families (1.2% on the fourth), which is what leaves the ~1.1% aspect error. Second, `Auto`'s under-correction is **magnitude-driven, not structure-driven**: it achieves 0.83–0.91 of ideal at 15°/6°, but 0.00 on the Manhattan control at 34°/12° — the single *wrong* cell in the set, where verticals get slightly worse. Earlier Manhattan-only runs overstated a structural story that is really about correction size, which is consistent with §3.5f's bimodality at high tilt.

Also confirmed independently: Auto and Full share one focal per photo — one cached solve, modes differing only in the damping applied afterwards. **Do not port a per-mode solver.**

Reproduce: `../evidence/non_manhattan.txt`, `nm_analyze.py`, `nm_metrics.json`.

### 3.6 A corrected reference implementation `[MEASURED]`

> **State of the port (§3.9).** The port is not usable on real photographs yet. It is a strong synthetic result, and the strength is policy, not solver. The two review defects are closed: the seed-4 `D_roll` failure was a family-selection bug — the lateral selector measured from the top-left origin and ignored the principal point, so a 19-support distractor outranked the real lateral family — and it is now 0.018° / 0.012°, worst 0.0503° over 80 seed sweeps; a wrong prior is recovered rather than trusted (1426 px against a 1400 px truth), held-out worst 0.0188°. **None of that is real-photo accuracy.** On 300 real photographs the port accepted 0 of 600 attempts and returned identity every time. Round 10 fixed the resolution-dependent gate (0.079° at 8653 px was unattainable) and acceptance moved to **80/900 attempts, 40 of 450 photos** — 8.9%, still a poor clone. The headline is **98% output policy and 2% solver**: oracle geometry under the same framing+scale policy scores 0.598% against the port's 0.587%, and both are in-sample. On real photos the port beats naive rectification 56.2% but **loses to plain identity 40.7%**, so it is a better naive rectifier, not yet a better correction than doing nothing. Round 11 showed the bottleneck is family selection, not detection (oracle selection cuts `Full` median 3.66% → 0.93%); round 12 built a VP/Mahalanobis-style selector and closed **zero** of that gap (3.6638% → 3.6720%), so nothing shipped and production was restored byte-for-byte. Round 13 then went back to the `Auto` damping that rounds 5 and 6 had failed on, and fixed it by changing the data instead of the form: 12,075 real Adobe pairs give a hump-shaped pitch response (peak ~0.80 at 6–12°, ~0 above 35°) where the port was monotone, and the shipped table moves `Auto` mean 32.35% → 27.76% and the strong scene 42.37% → 24.08% (target met) at a 2% `D_roll` cost. F1, F2, F3 and the 45° crash are fixed; F4–F13 are open. Treat 0.587% as a synthetic pipeline number, never as real-photo accuracy, and note it is `framing=True, scaling=True` only — `benchmark_adobe.py` reports the unframed ~31% path, so the two are not comparable. It also predates round 13. The docs are partly behind the code: README's framing claim, 75-test count and provenance rows are already corrected, but `port_status.txt` and `port_review.txt` still call the fixed seed-4 failure the top open defect, `OPERATIONS.md` still recommends a 154-line-stale `.bak` and a TSV header that is never written, and `doc_audit.txt` retains all 47 items.

With the measured constants above, `../upright_matched.py` fixes all three defects listed in §3.4 Result 6. It (a) solves the full rectification once, (b) recovers the true two-camera decomposition `H = s·K₁·R·K₀⁻¹` from that solve and then damps **only the rotation**, preserving `K₁ ≠ K₀`, and (c) applies the anisotropic saturating fractions of §3.5b. The round trip is exact (`2.8e-13` when the damping factors are set to 1), which is the regression test that the decomposition is being preserved rather than collapsed.

| case | mode | who | vertical conv. | horizontal conv. | crop kept |
|---|---|---|---|---|---|
| A_clean | full | Adobe | 0.73° | 0.36° | 77.1% |
| A_clean | full | **ours** | **0.49°** | **0.38°** | 94.8% |
| B_strong | full | Adobe | 0.98° | 0.23° | 68.0% |
| B_strong | full | **ours** | **0.96°** | **0.31°** | 100% |
| C_noisy | full | Adobe | 0.83° | 0.39° | 77.3% |
| C_noisy | full | **ours** | **0.50°** | **0.45°** | 94.9% |
| A_clean | auto | Adobe | 5.35° | 0.73° | 100% |
| A_clean | auto | ours | 3.08° | 0.46° | 91.4% |
| B_strong | auto | Adobe | 17.26° | 4.80° | 92.0% |
| B_strong | auto | ours | 14.58° | 2.67° | 81.1% |

`Full` now matches Adobe to within 0.25° on every scene (from 0.11° vs 0.73°, i.e. a 6× over-correction). `Auto` now lands in the right regime and tracks the tilt-dependent damping, though it is still consistently ~40% too aggressive and crops 9–19% where Adobe crops 0–8%. The modes now genuinely differ from one another, which they previously did not.

Reproduce: `../upright_matched.py`; results in `../evidence/matched_validation.json`.

### 3.7 Abstention and Guided — both resolved `[MEASURED]`

**Abstention — RESOLVED: it is not a veto, it is "nothing to correct".** Photos carrying an all-identity transform in every slot while still having `crs:PerspectiveUpright = 1` come out at **13.5% of the 17,669 photos holding any upright transform data**, or **9.6% of the 14,705 holding both the Auto and Full slots** (counts taken 2026-10-27; the catalog is live so they drift). The wider population is larger because it includes guided-only photos that have one key rather than a full mode set — quote whichever filter you mean, and prefer the both-slots figure for mode comparisons. Three layers could produce that — input (detector found nothing), solver, or output (a gate). The discriminator is the stored focal: a real `UprightFocalLength35mm` means the detector and estimator both ran; the default 35 means they never did.

A 17-image probe across four axes (noise σ 4–55, blur 2–11 px, contrast 5–45%, crop to 12% width) produced **zero abstentions** — every image yielded a real focal (31.6–40.8 mm) and a strong correction, with the Auto/Full magnitude ratio holding at 0.58–0.90. So none of those axes is the trigger, and it is not an output gate.

The decisive test is in the corpus. Comparing **Full's own** transform magnitude:

| population | n | Full \|H−I\| median | p90 | Full also identity |
|---|---|---|---|---|
| abstainers (Auto identity) | 1,404 | 0.0000 | 0.0000 | **100.0%** |
| corrected | 13,301 | 0.1657 | 0.6802 | 9.6% |

In **every** abstainer, Full is identity too. The solver decides *once per photo*, not per mode; when the required correction is negligible it writes identity into all six slots and every mode inherits it. Full's magnitude is a smooth continuum, not a step (p10 0.0004 → p90 0.6434), and the 9.7% of corrected-but-near-identity photos are simply the floor of that band where Full does ~0.0045 and Auto rounds to nothing.

So there is no confidence gate, no content test and no metadata signal **because there is nothing to gate**. The abstainers are not hard photos being abandoned — they are photos that are already upright. The 14% is a property of the photo population, not a hidden rule.

**The geometric mechanism is now known (§3.5g), and it softens the "no rule" claim above.** Abstention is not purely "nothing to correct": the solver enforces a hard minimum of **two distinct vanishing points**, and a Manhattan margin test. A scene with fewer than two VPs never reaches the solver at all. So a rule does exist — it is geometric, not confidence-based.

**Porting implication** is therefore twofold: emit identity when the solved correction falls below noise, *and* bail when fewer than two VPs survive with a 1.5× margin over the runner-up. Straight photos satisfy both conditions for the same reason.

**Guided is stricter than it looks.** It requires `UprightFourSegmentsCount == 4` *and* exactly four non-empty segments. Beyond the count rules, the guide *geometry* matters, and the degenerate shapes do not fail alike:

| guide geometry | behaviour |
|---|---|
| one contradictory guide among three good | **hard failure** — nothing written, three runs out of three. It neither averages nor picks three-of-four |
| all four collinear on the horizon | **hard failure** — nothing written. It does not infer pitch from horizon offset, and does not fall back to yaw-only |
| two near-parallel verticals | **solves and fits the guides exactly (0.000°), but under-corrects to 38% of the reference** — looks entirely successful |

So a guided success is not self-evident. Zero residual on the guides is a useless quality signal for degenerate input; check `max|H-I|` against a normal reference as well. Also note the failure marker: an **absent (`nil`)** slot 5 means the solve ran and produced nothing, while an **identity** slot 5 means it never ran.

**And guided is not additive.** After any real Auto or Full solve, a guided call **wipes slots 1–4 to nil** rather than leaving them. The earlier "writes slot 5 only" finding holds only on a photo whose other slots were already empty. Capture a prior result before invoking guided if you want to keep both.

The count-mismatch shapes:

| malformed input | observed behaviour |
|---|---|
| 3 segments, count = 4 | **nothing written at all** — no slot, no focal, though the Update click landed |
| 4 segments, count = 8 | **silently falls back to unguided**: guides discarded (count reset to 0), slot 5 left identity, and an ordinary solve written into slots 1–4 with a focal |

The second is the dangerous one, because it looks like success. Any harness or port must assert the count and all four keys, and verify slot 5 is non-identity afterwards. Evidence: `../evidence/guided_limits.txt`; n = 1 per failing shape, both deterministic. Untested there: contradictory, near-parallel, and horizon-only guides.

**`PerspectiveScale` is 100.0 for every single photo**, abstaining or not, and `PerspectiveRotate/Vertical/Horizontal` are all exactly 0. So the global Transform sliders are *not* where Upright writes; the whole effect lives in `UprightTransform_N`, and the Auto/Full crop is implied by the homography itself rather than by a subsequent scale. That matters for a clone: there is no separate crop stage to replicate, and no risk of double-applying one.

**Guided mode — RESOLVED (it is the strongest mode, and the cheapest to port).** No harvested photo used Guided, so it had to be driven by hand. The storage format was recovered from LR's own compiled Lua (`AgCameraRawControllerGuidedUpright.lua`): `UprightFourSegments_0..3`, each `"x1,y1,x2,y2"` at 9 decimal places in normalized frame coords, plus `UprightFourSegmentsCount`, with `PerspectiveUpright = 5`.

Trigger mechanics are the non-obvious part, and they cost several false negatives:

- `applyDevelopSettings` **can** write all of those inputs, and LR stores them verbatim — including the precise values echoed back from real drags.
- Writing them **does not run the solver.** No `UprightTransform_5` is produced and `UprightFocalLength35mm` stays at its default of 35. Three orderings and both pixel and normalized coordinate spaces were tried; all silent.
- The solve fires only on the panel's **`Update`** button (`addUprightGuide` → `_uprightGuidesDirty` → `flushDirtyUprightGuides`, all internal to the Develop module and unreachable from a plugin).
- Segments must lie inside `[0,1]` and be long enough to be usable; segments running off-frame are silently rejected.

With four guides drawn on true world edges of scene A_clean:

| mode | vertical | horizontal | p99 stretch |
|---|---|---|---|
| original | 15.52° | 5.32° | 1.000 |
| `Auto` | 5.35° | 0.73° | 1.111 |
| `Full` | 0.73° | 0.36° | 1.486 |
| **Guided** | **0.00°** | **0.00°** | **1.476** |

Guided is the only mode that reaches **exactly zero on both families**, and it does so at *lower* distortion than `Full` (1.476 vs 1.486). It writes slot 5 only and leaves slots 1–4 untouched.

**So Guided does not use the perceptual objective at all.** It takes the user's lines as hard constraints and solves the same `s·K₁·R·K₀⁻¹` family to satisfy them exactly. **This is the mode to port first**: no energy model, no line detector, no weights, no abstention heuristic — only the homography algebra and a solver.

---

**Mode semantics, confirmed at scale.** `Level` implies pitch and yaw of **0.004°** (i.e. exactly zero) while retaining 87% of the roll correction — it is a pure in-plane rotation, now verified on 12,883 photos and not just on synthetics. `Vertical` zeroes pitch (4.54° → 0.84°) *and* yaw (4.16° → 0.54°), which is physically necessary since yaw also makes verticals converge; its difference from `Full` is in the horizontal treatment, not the rotation.

**Abstention is mode-independent and solver-level**: an identical identity transform lands in *all four* mode slots (13.5% of 17,669 photos carrying any upright transform data) — matching the earlier finding that a single shared analysis is cached into every slot, and consistent with the §3.7 result that the decision is made once per photo.

Reproduce: `../analysis/corpus_lambda_curve.py`; raw output in `../evidence/lambda_curve.txt`.

---

### 3.8 Controlled-angle sweep: what `Auto` actually corrects `[MEASURED]`

Fourteen new synthetic scenes were generated with **pure, exactly known rotation** — seven with pitch only (5°…35°, yaw = roll = 0) and seven with yaw only (3°…24°) — imported into the catalog, and solved by the real Lightroom through the plugin's write-access path. Both families generate only the convergence they should: a pure pitch converges verticals and leaves horizontals exactly straight (measured `h₀ = 0.00°`); a pure yaw does the mirror image.

**Result 1 — `Auto` applies a steep, tilt-dependent damping to the VERTICAL family.** Vertical convergence removed, as a fraction of what `Full` removes, for pure pitch:

| pitch | 5° | 10° | 15° | 20° | 25° | 30° | 35° |
|---|---|---|---|---|---|---|---|
| `Auto` fraction | 0.810 | 0.882 | 0.787 | 0.842 | 0.565 | 0.295 | 0.078 |

Flat near 0.8 until ~20°, then collapsing: **half-effectiveness at ≈ 26°**, and essentially nothing by 35°. Fitting `λ/(λ+1)` with `λ = λ₀·exp(−θ²/2σ²)` gave `σ_v = 11.71°`, `λ₀ = 12.0` on this 7-point single-seed data (max err 0.107). **Superseded by §3.5f**: on 80 images the fit is σ_v = 13.33° keystone units / 22.26° pitch units, and λ₀ ≈ 2.6, not 12. Fitting a pure Gaussian instead is much worse (σ = 21.4°, err 0.214), so **the saturation is real** — the response is `λ/(λ+1)`, not `λ`. Model-free pairwise σ estimates drift from ~29° at low angle to ~11° at high angle, confirming a shoulder followed by a cliff rather than a single clean exponential.

The operationally meaningful constant is therefore **"half correction at 26°"**, not σ alone.

**Result 2 — `Auto` essentially does not correct PURE-YAW horizontal convergence.** This was the surprise, and it inverts a reading made earlier from one mixed-angle image:

| yaw | 3° | 9° | 12° | 18° | 24° |
|---|---|---|---|---|---|
| initial horizontal convergence | 3.00° | 8.93° | 11.82° | 17.42° | 22.67° |
| after `Auto` | **3.00°** | **8.92°** | **11.81°** | **17.41°** | **22.66°** |
| after `Full` | 0.48° | 0.49° | 0.48° | 0.87° | 0.88° |

`Auto` moves it by ~0.01° while `Full` removes 95–96%. So on a yaw-only frame `Auto` is almost a no-op. An earlier section inferred a large horizontal weight (`λ₀_h ≈ 51`) from a single image that had pitch 22° *and* yaw 6°; in that configuration the vertical correction itself drags the horizontal family along, which is enough to explain the observed 86% reduction without any yaw-specific term.

**This re-frames `Auto` as a vertical-only corrector**: level the horizon, fix the verticals with the damping above, and leave yaw-driven horizontal convergence largely alone. That matches its marketing ("upright" = make verticals vertical) and explains the 100% crop retention far better than a symmetric λ pair.

**Result 2b — RESOLVED: `Auto` is gated on PITCH.** The pure-yaw result above contradicted the mixed-angle frames, so an 11-scene follow-up sweep was run with a deliberate pitch bias: pitch fixed at 10° and 25°, yaw swept 0–24°. The gate is unambiguous:

| pitch \ yaw | 4° | 8° | 12° | 16° | 20° | 24° |
|---|---|---|---|---|---|---|
| **0°** (pure yaw) | — | — | **0.001** | — | — | **0.001** |
| **10°** | 0.949 | 0.806 | 0.798 | 0.724 | 0.716 | 0.413 |
| **25°** | — | 0.892 | — | 0.880 | — | 0.851 |

With *any* pitch present `Auto` corrects horizontals strongly; with pitch = 0 it does nothing at all. A 2-parameter gated model fits all 16 usable points to `maxerr 0.09`:

```
kh = saturate( L0h · exp(−yaw²/2σ_yaw²) · (1 − exp(−pitch²/2σ_gate²)) )
     L0h = 76.7   σ_yaw = 11.64°   σ_gate = 22.1°
```

**`σ_yaw = 11.64°` is a 3.0% match to the paper's `σ_h = π/15 = 12.00°`** — and to the float32 literal `1/(2σ_h²) = 11.3986332` found compiled into `CameraRaw`. The paper's `σ_h` is therefore real and governs the **yaw** component. The vertical constant is *not* settled to that precision — see §3.5f: it is unit-dependent (13.33° keystone units, 22.26° camera-pitch units, n=80) and the literal `π/12 = 15.00°` is bracketed in the first convention and excluded in the second. The horizontal figure here is unaffected by that ambiguity because yaw is the axis the weight actually keys on.

So the correct reading of the whole mechanism is **not** a single θ = total tilt with a λ pair. `Auto`'s two weights key on **different axes**, and the horizontal one is *switched off* when the verticals are already parallel. That is a far better explanation of the mode than anything in the paper: `Auto` is fundamentally a **vertical-driven** corrector which, once it decides to act, drags the horizontal family along with it.

**Result 3 — two anomalies and a caveat.** Y06 and Y15 are outliers where even `Full` failed. Both reproduce exactly, so they are deterministic solver failure modes, not noise: Y06 abstains silently (near-identity in every mode, 5.98° untouched) and Y15 applies a large warp that still misses (14.66° → 14.55°). Neither is predictable from metadata. Estimated focals across this sweep were also noisy (29.4–42.7 mm) because pure-yaw frames give the estimator no vertical constraint — the same degeneracy now quantified in §3.5f.

### 3.9 Thirteen rounds on the port: what closed, what did not `[MEASURED]`

The §3.6 callout summarises where the port stands; this section is the record behind it. Each round reproduced its failure before editing, and each was scored against the previous round's artefacts rather than a fresh baseline. Rounds 1–3 fixed real geometry defects. Rounds 4 and 6 changed what the metric measures, and in doing so changed what "fixed" means. Round 5 failed outright and was reverted. Rounds 7 and 8 recovered output policy. Round 9 took the port to real photographs and took the headline back. Round 10 repaired the resolution-dependent gate and did not fix coverage. Round 11 found that the bottleneck is family selection. Round 12 tried to close it and closed nothing. Round 13 went back to `Auto` with real corpus data and fixed it.

| round | target | outcome |
|---|---|---|
| 1 | VP family selection | **fixed** — distractor outranked the real lateral family |
| 2 | wrong focal prior | **fixed** — prior is now recovered, not trusted; held-out scenes added |
| 3 | stale post-reassignment VP | **fixed** — horizon term consumed an old subset-fitted VP |
| 4 | score against Adobe, not geometry | shape 7.36% better, raw 6.58% **worse** |
| 5 | `Auto` over-correction | **failed**, no change retained |
| 6 | what the two metrics actually measure | raw and shape disagree; raw is framing |
| 7 | output framing | 10% order-statistic recentre; raw 33.27% → 3.56% |
| 8 | output scale | cover/fit switch; raw 0.587% against a 0.403% floor |
| 9 | 300 real photos, frozen policy | **0/600 accepted**, all identity; 9.09% zoom step, 45° crash |
| 10 | the quality gate | resolution dependence fixed, coverage not — 0/600 (300 photos) → **80/900** (450 photos) accepted, median 2.076% |
| 11 | family selection vs input render | **bottleneck is selection, not pixels** — oracle selection cuts Full 3.66% → 0.93%; preview and radial correction rejected |
| 12 | VP / Mahalanobis-style selector | **closed zero of the gap** — 3.6638% → 3.6720%; nothing shipped, production restored byte-for-byte |
| 13 | `Auto` damping, measured from 12,075 real Adobe pairs | **fixed at the second attempt** — curve is hump-shaped in pitch, not monotone; Auto mean 32.35% → 27.76%, B 42.37% → 24.08% (target met), D_roll regresses 2% |

**Round 13 is the round-5 problem solved by changing the data instead of the form.** Rounds 5 and 6 both failed on `Auto` because all four cached scenes sit at 22° and 34° pitch — precisely where the port was already right. Decomposing 12,075 real Adobe `Auto`+`Full` pairs through the §3.1 model (4,966 surviving a pose cross-check against the `Vertical` slot) shows the true response is **hump-shaped in pitch**, not monotone: a shoulder of ~0.45 removed at 0–2°, a peak near 0.80 at 6–12°, a collapse to 0.16 at 27–35°, and ~0 beyond 35°, with a ~30% no-op branch still mixed in at 27–35°. The port over-removed below 5° by up to +0.32 and under-removed at 6–22° by up to −0.17, and 48.2% of real photos differed by more than 0.20. The units had to move for a reason worth recording: detected keystone compresses nonlinearly with pitch — scene A's true 22.0° is detected as 12.3° — so a keystone-units curve cannot express a pitch-units measurement. It ships as a 12-knot table of corpus medians times one calibrated scalar (`AUTO_PITCH_GAIN` = 1.20), clamped to [0, 1]; the fraction-to-scale inversion and the keystone target are untouched. `Auto` unframed mean drops 32.35% → 27.76% and shape 0.702 → 0.678; `B_strong` raw drops 42.37% → 24.08%, meeting the ≤25.38% target that had been open since round 4. `D_roll` regresses 0.58 percentage points (2%) because it is the only rolled scene and the curve has no roll coupling, and the strong band now under-corrects (−0.271) where it previously over-corrected (+0.548), halving the absolute error without reaching zero. The honest limits are that a ~30% no-op branch still survives at 27–35°, the argument may really be an internal rotation difference rather than camera pitch, and roll coupling is unmodelled. Evidence: `../evidence/auto_corpus.txt` (curve, controls, sensitivity); reproduce with `auto_corpus_measure.py` and `auto_corpus_verify.py`.

**Round 1** is the seed-4 failure. The detector emitted 711 `D_roll` segments; the real vertical and lateral families were both present with identical support counts at seeds 0 and 4, so support loss was falsified first. A spurious near-horizon VP survived with 19 supports after reassignment and won the lateral selector, which was computing `|vp_x| / hypot(vp_x, vp_y)` from the image origin — no principal point, no aspect, no focal. Its 0.999993 beat the real family's 0.996208. The distractor became the X axis, the real lateral VP was pushed into the depth slot, the starting yaw hit the optimizer's +68.755° bound and stayed there. Selecting in centred, focal-aware coordinates fixes it: 5.626° / 22.713° → 0.018° / 0.012°, and all 80 solves across seeds 0–49 and 0–9 land under 0.1° with a 0.0503° worst. The recorded bad output now returns `ok=False` rather than a plausible-looking wrong answer.

**Round 2** stops trusting the metadata focal. A 1200 px prior against a 1400 px truth previously produced 0.513° of horizontal error; the calibration now recovers 1426 px and 0.046°. The ±30% cases (980, 1820) reach the same solution. Round 2 also introduced the 14-scene held-out set that every later round is scored on, and its own worst was 0.1134° — the number Round 3 then chased.

**Round 3** found the horizon term consuming a VP fitted on a *subset* of supports before reassignment had expanded it. Family identity and supports were identical across seeds; only the stale VP moved, and it moved the roll/pitch/yaw compromise. Disabling the horizon term drops seed 0 from 0.1134° to 0.0116°; refitting the selected VPs on their final supports after identity is chosen gets 0.0122° without giving up the term. Two refit variants were rejected on measurement: TLS on every VP gave 18 bad candidates and 2 false passes; robustly refitting every unused VP before selection gave 30 and 26. Held-out worst falls to 0.0188° over 70 attempts, and 1,100 attempts all finish under 0.1° with no returned identity.

**Round 4 changes the oracle and the answer gets worse.** Scoring against Adobe's stored output instead of world geometry, with shape (similarity-invariant) and raw (untransformed) kept separate: shape improves 7.36%, raw regresses 6.58%, and `Level` sits at 1.4% raw because it barely moves the frame. The round's own verdict is that the port is closer to Adobe than naive full rectification and is not a clone. The raw regression is real and is framing, not algebra — the port fixes the destination principal point at the centre, Adobe does not.

**Round 5 failed.** The hypothesis was that `Auto`'s over-correction was a units error: σ_v is 13.33° in keystone units and 22.26° in camera-pitch units, and the port used 13.33 against camera pitch. Substituting the matching pair helps `B_strong` raw (42.37 → 36.61) and shape (1.9075 → 1.5087). Applying 13.33° directly to camera pitch instead goes further (23.47 raw) but gives back the whole shape gain (1.9483), and only works by restoring the units mismatch. Nothing was retained: production stays at 42.37 and the 25.38 target stays unmet. Round 5's useful output is negative — the units story is insufficient, and the four-scene corpus cannot separate a damping model from a framing error.

**Round 6** shows why. Holding the camera-pitch argument fixed, Euler scaling buys 1.21 raw points; narrowing σ then buys 11.92 more and *reverses* the shape gain (1.4492 → 1.9483). `A_clean` does the same thing at a different scale. So a raw win can be bought by under-correcting, and raw is dominated by where the output lands, not by how straight the lines are. That reorders the work: framing first, damping later.

**Round 7 recovers the framing policy.** Per output axis, sort the four transformed source corners, take the outer pair and the inner pair, and blend 10% outer / 90% inner; centre the blend midpoint on the frame. For `c = (s₀ + 9s₁ + 9s₂ + s₃)/20` the translation is `(.5,.5) − c`. All 16 active Adobe slots already satisfy that identity to the 9-decimal storage precision, with rule errors of 1e-10 to 1e-6 px, and it is not bbox recentring, centroid anchoring or crop-origin zeroing. Shape is unchanged to 1e-12 because a translation cannot change it. Raw collapses: 33.27% → 3.56% pooled, with the four `Auto` slots going 28.7–42.4% → 0.85–2.74%. `Level` is unmoved at 1.4% — it has no framing translation to recover.

**Round 8 recovers the scale policy.** Cover the frame unless covering crops more than 10% of the corrected quad, in which case fit the inner bounds instead — implemented as a ratio test at 1.10 between the cover scale and the inner-bounds scale. Raw on 160 paired attempts goes 33.27% → 3.56% → **0.587%**, against oracle floors of 0.430% (scale alone) and 0.403% (scale plus translation). All 16 slots reproduce to a 5e-6 scale RMSE. The rejected alternatives are informative: fixed `PerspectiveScale`, area preservation, fixed quad area, fixed crop fraction, and a single always-active displacement cap all fail, including per-mode constants. **The official benchmark measures the other path**: `benchmark_adobe.py` calls `solve_ex` with no `framing`/`scaling`, so it reports ~31% and never sees the 0.587. A framed number and a benchmark number are not comparable.

**Round 9 takes the port to real photographs, and it returns nothing.** 300 photos, uniform sample without replacement from the 5,443 available qualifying paths (seed 9009; 255 RAW, 33 TIFF, 12 JPEG), detection at the default 1200 px and supports rescaled to original dimensions, one seed, both `Auto` and `Full`: **0 of 600 attempts are accepted.** Every returned `H` is exactly the identity, so framing and scaling change no real-photo score. The rejection split is 444 "poor undamped family alignment" (222 photos) and 156 "insufficient line structure" (78 photos), across every format and orientation. A separate JPEG run through `solve_ex(image=…)` at full resolution, bypassing the benchmark's support-rescaling seam, also returns 24 identities.

The cause is a gate that was never sized for real files. Round 1 added `_alignment_quality` to stop exactly the seed-4 class of failure, and calibrated the threshold as three times the existing four-pixel loss scale: `limit = min(1, 3·s)` with `s = 4/(max(w,h) − 1)`. That is an explicit choice, not a recovered rule, and it was justified by headroom measured on 1400×1000 fixtures — 0.491° there, with the worst passing median at 0.127°. At 8653 px the same formula gives **0.079°**, and a real photo's line medians sit far above it, so `scatter_passed` fails for essentially everything and the solve is thrown away as bad alignment. The fixtures could not catch it: they are small, and the gate was tuned against them at exactly that size. It still rejects the seed-4 fit correctly — the recorded bad output returns `ok=False` — but on real files "correctly rejected" is the whole corpus.

Round 9 also breaks the scale policy on its own terms. The 1.10 switch is a **9.09% zoom discontinuity**, not a steep slope: at ratio 1.10 the rule selects 1.10, one step above it selects 1.00, and the jump survives as the geometric perturbation goes to zero. Equality is numerically unstable — pre-framing an already-centred matrix flips a rounding bit and changes the branch, as does a 1e-15 translation. Separately, `scale_output` raises an unguarded `ZeroDivisionError` on an exact 45° rotation: 20 of 144,144 family trials, with `H = [[1,−1,.5],[1,1,−.5],[0,0,1]]`, where the inner bound collapses to `.5/0`. That matrix is a valid convex, positive-determinant, horizon-free map. `solve_ex` catches `ValueError` at that call and not `ZeroDivisionError`, so it escapes to the caller. A native-pose control (44 of 600 slots over 5%, worst 26.5%) further shows the two-boundary rule is not broadly equivalent to Adobe's scale. That crash is now fixed: `scale_output` falls back to the cover scale when the inner bound collapses and returns 1.0 on that matrix. The 1.10 switch discontinuity itself is unfixed; 102 tests pass.

**Round 10 re-sizes the gate, and coverage barely moves.** The old limit `asin(min(1, 12/(maxdim−1)))` was 0.491° at 1400 px and **0.079° at 8653 px** — an optimizer noise scale, not a resolution-independent test. Round 10 replaces it with per-family pixel-angle statistics: weighted median absolute angle `a`, weighted median signed bias `b`, effective count `n_eff`, and `u = 3.72·MAD/√n_eff`, with both families required to satisfy `a ≤ 1.0°` **and** (`b + u ≤ 0.1°` or `b + u < a`). The statistics are invariant to uniform resizing, so pixel dimensions leave acceptance entirely. Acceptance on 450 Adobe-corrected photos and 900 `Auto`+`Full` attempts goes **0/600 → 80/900** (8.89%, 40 of 450 photos); median 2.075952%, p90 7.979433%, max 81.961591%. That is not a usable clone. Precision on the accepted set is 97.5% at ≤5% error, but recall is 19.75% of fitted candidates, and the two false accepts are related DNG copies scoring 5.70/5.73% against 3.37% for identity. The gate is conservative on purpose: the 1° cap was the largest tested with zero accepted >5% on calibration, and the 1.5° alternative admits a 41.1% error. Abstention is preserved — 0 of 456 Adobe-abstained records are accepted — but 208 of those 456 never reach the gate anyway. Synthetic output is untouched: 640 sweep maps and 280 held-out maps are bit-for-bit equal, and the suite goes 99 → 101 tests.

**Round 11 shows the bottleneck is family selection, not detection.** The intervention reuses the cached round-10 detections exactly and changes only VP/family selection: each segment is labelled by its angle in Adobe's own `Full` output, within 1° of vertical or horizontal, ≥3 per family. On all 387 RAW files, `Full` median goes **3.663819% → 0.932318%** (74.55%) and `Auto` 2.061794% → 0.874546%; on the same 361 eligible files, `Full` goes 3.550164% → 0.873930%. A no-depth control sits at 3.663458%, so removing the depth term does not explain it. Better pixels do not help: an embedded-preview path *worsens* `Full` median to 3.827099% and leaves the gate at the same 74/766 accepts, and every tested generic radial correction (both signs, 5% and 10% corner radius) worsens `Full` median while acceptance falls from 14/80 to 8–12/80. The oracle is a diagnostic, not a deployable classifier — it uses Adobe's target to label axes, and 40 attempts still exceed 5%, four above 20%. It bounds the headroom; it does not hand over a selector.

**Round 12 builds the selector and closes nothing.** A full frame selector — centred isotropic duals, dual-plane consensus hypotheses, disjoint minimum-residual assignment with a 1.5° ceiling, joint frame ranking with an inferred third axis, both margin inequalities, focal-aware orthogonality and three assignment/refinement passes — does improve every intermediate it was aimed at. Length-weighted V/H purity rises 34.41/21.37% → 65.16/46.74%, horizon failures fall 70 → 10, valid `Full` maps rise 214 → 308, accepts 37 → 57, and the median Euler-vector norm drops 26.03° → 19.30°. The number that matters does not move: **`Full` median 3.663819% → 3.671977%**, p90 14.615070% → 17.581447%, and the oracle gap closed is **−0.2987%** — negative. On the 189 files with valid maps in both versions it is 2.576683% → 2.602056%, so denominator growth is not the explanation. The candidate also accepts a new bad `Auto` result on the abstention control (id2119129, 6.446840% against identity), and `Full` still loses to identity more often than it wins (decisive 47.08%). Synthetic Adobe agreement improves slightly (0.587092% → 0.582877%, framed+scaled path) and is overruled. Nothing ships; production is restored byte-for-byte to the same SHA256, and the deployable result remains round 11's baseline.

**What is honestly still open.** Real-photo acceptance is the top item, and round 10 moved it by 8.9% of attempts. The port still corrects almost nothing on real files, so **0.587% is a synthetic-only figure and must not be quoted as real-photo accuracy**. It is also a `framing=True, scaling=True` number, and it predates round 13's `Auto` curve; the official `benchmark_adobe.py` reports the unframed path (~31%), so a benchmark number and this one are not comparable. The unconditional real-photo verdict is worse than the accepted-subset numbers suggest: on 329 paired records the port beats naive full rectification 56.2% but **loses to plain identity 40.7%**, with medians port 2.201% / naive 3.350% / identity 2.023%. Its entire advantage is `Auto` (86.1% win over naive); `Full` wins only 25.8% against naive and 39.3% against identity. The port is a better naive rectifier, not yet a better correction than doing nothing. The tail is Adobe's own correction, concentrated in `Full`, portrait orientation and 35–40 mm lenses, and the gate's 1.0° separation is what keeps the abstention control clean.

**The family axis does not break it; pose does.** At the 15°/6° family pose all four non-Manhattan families correct as well as the Manhattan control in `Full` (0.2–1.0 px similarity, verticals 3.3–4.3° → 0.00–0.02°), and the control's own spread covers the ordering. There is no family-specific defect at any pose. The failures are mode- and pose-driven: `Auto` collapses above 25° pitch/yaw — at pitch 40 it is at −0.13…0.03 of ideal, at yaw 30 it is *worse than identity* (−0.69…−0.43) — and `Full` breaks on the two scenes containing a floor at pitch ≥40 under a wide lens. There the floor's depth lines spawn a spurious lateral VP farther out than the true one, "largest lateral" picks it, and pitch overshoots 40° → 52.5°; `CT_manhattan2` ends at −1.22 of ideal, worse than identity, and `NM_three_plane` at 0.83. At 1600 px the same pose is fine (4.6 px), so it is wide-lens + floor-plane, not pitch alone. The gate refuses correctly in 18 of 20 real refusals and misses that one bad solve: the relaxed bias test passes any bound below the family's own scatter (0.257 against a 0.649 median) even though the rotation is 12.5° wrong. It also false-refuses two good cells (`NM_curves` at yaw 35/900 px, candidate 0.965 of ideal). Focal calibration accepts 8–140% error at the extreme pitches, and default framing leaves 0.0–0.2% of the source in frame at pitch ≥40 — a fill failure, fixed by `scale_output`, not by the gate. One positive worth keeping: the port does **not** straighten arcs, matching Adobe. A 7 m arc goes 1.005 px rms → 0.026 px with the fitted radius 349.9 against a true 350.0 and aspect 1.0004 — there is no straightness prior, and Adobe also holds the radius near 350 (`non_manhattan.txt`). All seven featureless frames refuse with exact identity, so the solver does not invent a correction where the gate is doing the work.

**F1 — `ok=True` while the frame crosses the projective horizon (fixed).** In the verified build `f2a900d9`, `solve_ex` validated finiteness but never checked corner `w`, so automatic and guided modes could return a homography that puts one to three frame corners at `w ≤ 0`. The corners then divide into a wrapped quad that cannot be sampled as a frame, and `solve()` returned the same H with no signal at all. The check already existed inside `frame_output`; it was simply not applied unless `framing=True`, and with `framing=True` the same inputs correctly returned `ok=False, reason="framing crosses the projective horizon"`. The rate was not exotic: 24 of 320 `ok=True` results straddled in a random-scene sweep, and 96 of 500 realistic guided quartets (19%). In the fuzz matrix 34 of 82 `ok=True` results were unusable. `level` was the only mode that never straddled, because it applies roll only. Reproducer against the verified build (`focal_px=900` is an 83° hfov, nothing exotic):

```python
import numpy as np, upright_port as up
W, H = 1600, 1200
seg = np.array([[564.31,685.56,351.83,1548.71],[718.24,677.62,719.95,1368.26],
                [847.02,670.98,970.79,1245.29],[353.63,738.6,967.66,693.59],
                [258.88,914.01,1009.01,796.95],[97.22,1213.29,1066.68,941.1]])
for mode in ("vertical", "full"):
    out = up.solve_ex(seg, w=W, h=H, focal_px=900., mode=mode)
    C = np.array([[0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]])
    print(mode, out["ok"], (C @ out["H"].T)[:, 2])   # ok True; z straddles 0
up.solve(seg, w=W, h=H, focal_px=900., mode="full")  # silently the bad H
```

On `f2a900d9` both modes return `ok=True` with a straddling quad (vertical `z = [1, 1, −11.7, −11.7]`). The fix is now in `solve_ex` unconditionally: the same reproducer returns `ok=False, reason="crosses the projective horizon"` and `framing=True` agrees. Two of the other fuzz classes are also fixed. **F3** is fixed: `_classify` and the guided axis test compare in pixels instead of normalized coordinates, so 20:1 frames no longer classify every segment as vertical — a wide frame now reaches the solver instead of dying at `insufficient line structure`. The **45° `ZeroDivisionError`** in `scale_output` is fixed too: a collapsed inner bound falls back to the cover scale and returns 1.0. **F2** is fixed at all three `max(w,h)−1` divisors (`_alignment_quality`, `_family_terms`, `_refine_vp`), so a 1×1 image now returns `insufficient line structure` instead of raising. The remaining classes are open: **F4** `scale_output` does not commute with `frame_output` and leaves a 1e6 input translation untouched; **F5** `scale_output` rejects any mirrored quad while `frame_output` accepts singular input silently; **F6** all-NaN output on `H22` underflow; **F7** `guided` returns a rank-deficient H with `ok=True` at `focal_px=1e12`; **F9** no detector output for images up to 4×4 and 1.0–3.4 GB at high resolution; **F10** 100k segments find no VP; **F11** `guided` raises where the rest returns `ok=False`; **F12** NaN segments are dropped with RuntimeWarnings; **F13** `mode="off"` still runs the detector. 102 tests pass.

Reproduce: `../evidence/round1.txt` … `round13.txt`; real-photo verdict `real_photos.txt`; family/pose stress `port_families.txt`; fuzz and F1 `port_fuzz.txt`; Auto corpus `auto_corpus.txt`; port tests via `python3.12 -m pytest -q port`.

### 3.10 Independent verification: the headline is 98% policy, 2% solver `[MEASURED]`

A separate pass re-derived every headline number without importing the port's own benchmark code (`evidence/verify_metrics.py`, `verify_seeds.py`, `verify_perturb.py`). The arithmetic reproduces exactly — 33.266029 → 3.562255 → 0.587092, and the 0.403312 shape bound — so the figures are not fabricated or mis-transcribed. Three of the framings around them do not survive.

**The solver is at oracle level, no better.** Applying Adobe's own framing and scale policy to a naive full rectification at the true rotation gives 3.1071% mode-blind; with mode semantics and perfect rotation it gives **0.5980%**, against the port's 0.5871%. The port is 2.4% better than oracle geometry under the identical policy. Per mode (port / naive+policy / oracle-rotation+policy): auto 0.9224 / 3.2891 / 0.9331, full 0.1986 / 0.2103 / 0.2103, level 0.9692 / 4.3963 / 0.9848, vertical 0.2581 / 4.5329 / 0.2755. The policy alone accounts for **98.2%** of the raw reduction (33.266 → 0.587); geometry accounts for the rest. And 0.587 needs `framing=True, scaling=True` — the bare solve is 33.27%, so it is a pipeline number, not a solver number. On `Full` alone the port beats both controls by 5.9%, which is the one mode where the solver is doing visible work.

**The 0.587 is a framed+scaled number and the official benchmark is not.** `benchmark_adobe.py` calls `solve_ex` with no `framing`/`scaling`, so the benchmark measures the unframed path, reports ~31%, and never sees the sub-1% figure. Every 0.587-class number here is `framing=True, scaling=True`; every benchmark-class number is bare. Comparing one to the other overstates the gap by roughly 50×. Round 13 also post-dates this verification and changes the `Auto` response, so 0.587 is now a pre-round-13 figure — re-run it before quoting it as current, and re-run `benchmark_adobe.py` only for the unframed comparison.

**Both headline numbers are in-sample.** The 10/90 framing weights and the 1.10 scale switch were fitted on the same 16 slots plus the 320 sweep slots, and round 8 already recorded that the sweep was not a blind holdout. There are only ~40 distinct `H` behind the 80 attempts (2–3 unique per 5 seeds; `A=C` and `D=A+roll`, so two base poses, effective n ≈ 40). The 0.403312 bound is the best translation+uniform-scale alignment of the port's *own* unframed `H`, not a solver floor, and it is computed from the same `H` it is compared with. The held-out set is frozen but no longer blind: the generator's mtime predates the first score, yet round 3 tuned against H09 and every later round re-ran it, so 0.0188° is post-hoc.

**The seed fix and the perturbation result both hold.** A fresh process, 200 attempts over seeds 0–49: worst 0.0488° (`B_strong`), zero over 0.1°, zero rejected. `D_roll` seed 4 is 0.013721/0.001034 and no longer an outlier; seeds 10–49 are no worse than 0–9, so there is no seed cliff; replaying the recorded bad theta through the public API returns `ok=False` and exact identity. Perturbations also hold: JPEG q75, blur σ0.8, contrast 0.85/1.20 and resize 0.75×/1.5×, 140 scores each, worst 0.0270°, zero over 0.1°, zero rejected. Mean error degrades 1.66× under downscale and still sits 3.7× inside the 0.1° gate.

**The "1e-6" scale claim is 1e-5 as a coefficient.** Independent reconstruction of the native maps gives max 9.882e-6, RMSE 4.975e-6, mean 3.363e-6 on the scale coefficient itself. It is 1e-6 only when quoted as a fraction-of-diagonal raw distance (mean 1.083e-6, max 3.078e-6). The 16 slots are the fit slots, so this is fit quality, not a held-out result, and `V40S3` `Full`/`Vertical` still fail by 0.0466 scale (3.96% raw).

A transient round-12 prototype scored 0.582340 against the stable file's 0.587092; the numbers here are the stable `f2a900d9` version and predate round 13's `Auto` curve. The honest summary is the one the verifier wrote: *port + a policy fitted to Adobe reaches 0.587% in-sample, versus 0.403% if the same map is given the oracle scale; oracle geometry + the same policy reaches 0.598%, so the solver contributes ~2% and the policy ~98%.*

## 4. So what is "special"? (and what is not)

| Component | Status |
|---|---|
| line extraction | standard-ish (`[BINARY]` is bespoke a-contrario, not LSD; `[OSS]` darktable uses LSD) |
| VP voting | per-segment line dual + vote counting (`[BINARY]` §3.5g — not RANSAC) |
| Manhattan frame from 3 orthogonal VPs | textbook |
| Powell / direct-search optimisation | standard |
| **`H = s·K₁·R·K₀⁻¹` — a rotation re-projected through a *different* camera** | **Adobe-specific, and the deepest secret** `[MEASURED]` |
| **Joint `estimate_MKR` instead of sequential** | **paper contribution** |
| **`E_eye` horizon term, line-weight normalised** | **Adobe-specific** |
| **Adaptive `λ(rotation)` — the "looks right" knob** | **Adobe-specific, the key idea** |
| **Perkins's-law inequality + log-barrier** | **uncommon**, borrowed from perception literature |
| **`E_img` Jacobian penalty on curved edges** | **uncommon** |
| **`E_f` focal-length plausibility** | **uncommon** |

**The one-sentence version:** *the novelty is the objective function, not the geometry.* Lightroom's Auto is a **weighted, perception-regularised compromise between "rectify" and "leave it alone"**, where the compromise itself depends on how tilted the camera was. A conventional "detect VPs → push them to infinity" implementation is not a clone; it is the `Full` mode with the safety rails removed.

**And the part that is genuinely non-obvious even to experts:** the output is `s·K₁·R·K₀⁻¹`, *not* `K·R·K⁻¹`. Everyone who guesses at this guesses the rotation homography with a shared `K`, which is off by 10.6% on real photos (32% on synthetic). Shipping two independent intrinsics is what lets Adobe make a wall parallel without simultaneously forcing the perspective recovery a true physical rotation would. That is the single highest-value line in this document — though see §3.1 for what the 1e-15 does and does not establish.

**But be honest about the limits of the claim:** on clean, well-textured, exactly-Manhattanish input the perceptual terms bind much more weakly — but they do **not** vanish. Measured against the real solver (§3.4), `Auto` still differs from `Full` by 4.6° of residual vertical convergence and a 23% crop, on a *clean synthetic* scene. The differences get larger on real photos (noise, partial line sets, residual lens distortion, non-Manhattan content). Our own prototype, however, currently collapses all four modes to one homography — so "the modes differ only on real photos" was the wrong explanation; they differ always, and our implementation is what is failing to reproduce it.

---

---

## 5. Patents — the practical blocker (live until 2033)

> **Read §5a first.** A claim-by-claim mapping against the shipped product overturned three assumptions, including the one this report originally led with.

### 5a. What is actually claimed `[BINARY + web]`

**Correction 1 — US 9,088,885 is not in this family.** It is "Implicit registration for multiple identities" (telecom; MDN/HSS/IMS). The family member is **US 9,098,885** only.

**Correction 2 — the adaptive weights are not claimed at all.** `σ_v = π/12`, `σ_h = π/15` appear in paper Eq. 5 and in all five specifications as Eq. A5, and in **zero of the 100 claims**. The shipped `0.2618f`/`0.2094f` literals are just those published numbers sitting in an unclaimed slot. Changing the σ values, changing the decay shape, or dropping the weights entirely avoids no claim. (They remain essential to *matching output* — see §3.5f — just not to clearance.)

**Correction 3 — VP selection is covered only functionally.** No claim recites the iteration count, the ≥3 inlier floor, τ = √3, the candidate cap, or the tournament. The covering language is 9,098,885 claim 1: "determining initial values for an energy function according to the extracted line segments." A different detector or a different RANSAC does not design around it.

**The real exposure is behavioural, not structural.** US 9,008,460 claim 1 and US 9,582,855 claims 1 and 11 turn on the phrase *"homography … computed at least in part according to one or more human perception-based criteria."* No claim 1 in the family recites a detector, a solver, or any named term. It reads on behaviour — any transform computed under a criterion that trades full rectification against looking wrong. **`Auto` is exactly that behaviour**, so shipping an `Auto` equivalent re-exposes 460 and 855.

**The one clean design-around is Guided.** "user", "guide" and "guide line" appear in none of the 100 claims, so guided clears 9,098,885 c1, 9,008,460 c1, 9,727,787 c1 and 9,098,885 c1 on their face — and it produces better output anyway (§3.7: 0.00° on both families at *lower* distortion than Full). Residual hole: 855 c11 is a reprojection-free "planar transformation … according to human perception-based criteria" claim, and a court could call user-drawn world edges perceptual. Mitigate by keeping the guided solve purely algebraic and never describing it in perceptual terms.

**Published vs patented-only.** The two-intrinsic H, all four energy terms, the Perkins fork rule, the f = W initialisation and the joint MAP are all published, so they are citable prior art against others but not free to use. The only genuinely new matter is the Atlanta-world second orthogonal frame (9,098,885 c1/c12, 9,727,787 c1). The product only partially practises it — free VP clustering, a 9-candidate cap, a tournament, no explicit second frame — yet still corrects non-Manhattan scenes, so "we skipped Atlanta" is an argument, not a proof.

**Two places the product does not practise its own claims, both free to copy**: 954 c6 (focal-relationship energy term — the product has no prior weight at all, metadata is a discrete switch, §3.5d) and 954 c17 (init requiring f = image width — the product falls back to 35.0 and clamps metadata only to [10, 1000]).

**Do not design around the two intrinsics** (460 c6/c13, 954 c5/c6) — a shared K is 10.6% off on real photos. Keep the mode semantics; 855 c14 claims Level and Vertical by name.

Full claim text and per-limitation mapping: `../evidence/patent_map.txt`. Engineering guidance, not legal advice; no prosecution history, doctrine of equivalents, or foreign counterparts read. Nothing is clean before 2033-04-26.

---

`[PATENT]` A five-patent US family, assignee Adobe Inc., inventors Shechtman / Jue Wang / Hyunjoon Lee / Seungyong Lee, shared provisional priority **2012-04-27**, all **Active**:

| Patent | Covers | Expiry |
|---|---|---|
| [US 9,008,460 B2](https://patents.google.com/patent/US9008460B2/en) | camera info → homography **under "human perception-based criteria"** → reproject (method) | 2033-05-14 |
| [US 9,582,855 B2](https://patents.google.com/patent/US9582855B2/en) | **same, apparatus claim — broadest, no line/VP limitation** | 2033-06-25 |
| [US 9,098,885 B2](https://patents.google.com/patent/US9098885B2/en) | calibration via multiple non-aligned orthogonal frames | 2033-04-26 |
| [US 9,519,954 B2](https://patents.google.com/patent/US9519954B2/en) | line segments + energy-function optimisation of intrinsics/orientation/VPs | 2033-04-26 |
| [US 9,729,787 B2](https://patents.google.com/patent/US9729787B2/en) | line segments → orthogonal frame → eye-level estimation | 2033-04-26 |

- The perceptual criteria themselves are *dependent* claims 10–12 of `US 9,008,460` (picture-frame alignment, eye-level alignment, image-distortion constraint), not a standalone claim.
- Because the qualifier is **functional** ("human perception-based criteria"), swapping the line detector, the optimiser or the VP solver does **not** design around it.
- **Nothing relevant expires before 2033.** Only A2iA/ABBYY `US 8,885,972` (2028) is near-term, and it is document-oriented.
- Adjacent/less relevant: US 11,393,072 (Adobe ML auto-rotation, 2041), US 9,478,006 (Adobe content-aware **crop UI**, 2035), **US 9,064,309 is Google** (not Adobe), ABBYY family (2028).

**Consequence:** the algorithm is open and published, but the **method is patented**. Shipping a clone in 2026 means either a licence or accepted infringement risk. This is the thing most people get wrong when they say "just reimplement the paper."

---

## 6. Existing open-source prior art

`[OSS]` Full detail in `handoff/auto-upright-survey.md`. Headline:

- **darktable `ashift`** — the only production-grade FOSS auto-keystone. **GPL-3.0-or-later**, and it vendors **AGPL-3.0 LSD**. Detection is LSD + a 60 %-elimination RANSAC (`_remove_outliers`), Nelder-Mead on a point-to-line cost, rotation + keystone + shear, **no focal estimation**. Closest thing to `Vertical`/`Full`; has no real one-click `Auto`.
- **RawTherapee** vendors `ashift.c` **verbatim** (`rtengine/ashift_dt.c`) — so it is one implementation counted twice, not an independent data point.
- **Hugin `linefind`/`FindN8Lines`** — GPL-2, Canny + 8-neighbour chaining, and one genuinely good idea worth copying: it **scales the straightness tolerance by focal length in pixels** so lens-curved edges are not mistaken for bent lines.
- **OpenCV** (Apache-2.0): LSD is in `imgproc` (no contrib), plus `findHomography`, `decomposeHomographyMat`. **Verified false:** opencv_contrib does *not* ship a `vanishing_point`/`VPSolver` (checked full tree on 3.4/4.x/5.x).
- Permissive VP solvers that do exist: `xiaohulugo/VanishingPointDetection_BSD2` (BSD-2), `rayryeng/XiaohuLuVPDetection` (MIT).
- Permissive DL: `cvg/VP-Estimation-with-Prior-Gravity` (MIT, needs gravity prior), `cvg/GeoCalib` (Apache-2.0, supplies gravity + focal), NeurVPS (MIT, stale), DeepLSD/pytlsd (MIT).
- Closest Lightroom-specific clones: `chsasank/Image-Rectification` (BSD-3, Google's ICIP-2014 method), `metamountain/Perspective-Correction` (MIT, has a confidence gate).

**Gap: there is no MIT/Apache end-to-end Upright clone.** Everything permissive is either bare primitives or research code.

### 6a. NegPy, measured against this `[MEASURED]`

NegPy (`/Users/rohan/Projects/negpy`, `lab-redesign`) is the closest FOSS analogue in spirit — it auto-corrects geometry from detected film/gate edges rather than picture content. It is solving a different problem (scanned film, where the gate *is* the ground truth), so this compares the models, not the products.

**What its warp actually is.** `keystone_quad` displaces all four unit-square corners by the *same* vector `(cv,ch)·0.005` in a checkerboard sign pattern. That is a genuine homography, but a **2-DOF slice** of PGL(3)'s 8: the centre and all four edge midpoints are fixed, so the principal point is nailed to the frame centre and no asymmetric trapezoid is representable. `|Δx| = |Δy|` at every corner, so the axes are coupled. Measured structurally, one axis alone sends its own VP finite and the other's to infinity — the family can make at most one line family parallel, and only by zeroing the other parameter. Any non-zero `ch` tilts the horizon 31° while the true roll-0 horizon is flat. It carries an *implied* focal (`13397/cv` px) that is never estimated: the shipped `cv = 5` implies 2680 px against a truth of 1600.

**At 15° pitch / 6° yaw** (metric calibrated by reproducing this report's own 15.52° scene-A figure), original vertical convergence 10.77°:

| | vertical | horizontal | note |
|---|---|---|---|
| Guided | 0.00° | 0.00° | |
| Lightroom `Full` | 0.51° | — | |
| NegPy, uncapped | **0.09°** | 2× worse than Full | **beats Full vertically**, 5.0% of frame width off the true quad |
| NegPy, shipped 5% cap | 4.29° | — | 60% removed, 41% of the correction clipped |
| NegPy live solver | 10.63° | — | dropped `cv` entirely; the −1.6° rotation made the dropped axis *worse* than untouched |

**The finding that reframes it: the family is not the limit, the cap is.** Uncapped, it beats `Full` at every angle from 5° to 40°. Capped at `_MAX_KEYSTONE = 5`, it is a ~5%-of-frame corrector that degrades to near-worthless past ~22° and is marginally *harmful* at 30°. `cv` saturates at ~9° pitch. Its manual sliders do not rescue this — they cap at ±15%.

**Where the film-edge approach genuinely wins:** content-independence, semantic structural confidence gates (`from_film_box` / `has_opposite_pair` / `at_canvas_edges`) rather than magnitude abstention, determinism in ~12 Gauss-Newton steps with no RANSAC or VP voting, the gate doubling as the crop rectangle, and no patent exposure at all. It fails with no visible gate (correctly abstains, but that is zero coverage on Lightroom's population), on tight scans that lose the vertical pair, and its `_MIN_KEYSTONE = 0.3` / `_KEYSTONE_GAIN = 2.5` discard real small corrections on clean frames.

**The honest analogue for NegPy is Guided, not `Full`** — both are constraint-driven, but NegPy's constraint set is only 2-DOF, so it reaches 0.61/0.42 where Guided reaches 0.00/0.00.

**Upgrade path that keeps the philosophy:** treat focal as a *camera* property. Swap `_solve`'s `(angle, cv, ch)` for `(r_x, r_y, log f)` keeping the same residual, Jacobian and gates; add a log-`f` prior from EXIF or format nominal; add `rotation_quad()` beside `keystone_quad` (`keystone_matrix_normalized` already derives everything from it); replace `_MAX_KEYSTONE` with degree limits; keep `cv/ch` in config and migrate by fitting the old quad at a default focal. **Two-line stopgap that buys most of it: `_MAX_KEYSTONE` 5 → 20 and the sliders ±15% → ±30%.**

Reproduce: `../evidence/negpy_compare.txt`, `analysis/negpy_compare_calc.py`. Caveats: one synthetic family, roll 0, Lightroom side ratio-matched rather than re-run.

---

## 7. Delivery & failure modes (where a clone wins)

`[ADOBE-DOC]`/`[RE]`: the solver runs as a **discrete Develop-time analysis** and writes a cached 3×3; the renderer only applies it. Modes are precomputed into enum-indexed slots. A ≤110 % auto-fill cap means beyond that it leaves gaps. Adobe's own error string is *"No upright correction found."*

Known failure modes (all sourced in `LIGHTROOM-UPRIGHT-FORENSICS.md`):
1. **Gaps are never filled** — Adobe's #1 complaint since 2016; deferred to Generative Expand.
2. **Corrects confidently on noise.** This was originally written as "abstains on non-Manhattan scenes", and §3.5h overturned it: single walls, railings, three-plane corners and curved content all correct to 0.83–0.96 of ideal, so it does *not* need Manhattan geometry. The failure mode that survives is the opposite — on a noisy or near-featureless frame the ≥2-VP gate is the only thing standing between you and a confident bad correction.
3. **Deliberate over-correction cap** — users report results changing between releases (`UprightVersion`).
4. **Crop destruction** — switching mode silently resets crop and prior Transform settings.
5. **Non-determinism** — Adobe admits the result varies with content/lighting; digest invalidation means identical sliders can re-render differently.
6. Requires lens correction first; homography cannot fix fisheye.

**Three concrete differentiators for a clone:** (a) content-aware/outpaint the gaps instead of cropping, (b) a real confidence gate that abstains cleanly, (c) deterministic, reproducible solves.

---

## 8. Recommended build (license-clean, no patents copied verbatim)

**Order changed by §5a: build Guided first, Auto last.** Guided is the only mode that escapes the behavioural claims on its face, *and* it is the strongest mode with the lowest distortion. `Auto` is the expensive one — it is the exact behaviour the live claims read on.

Tier 1 (no energy model, no perceptual language, no claim surface): guided solve of user segments as hard constraints over `H = s·K₁·R·K₀⁻¹`. Pure algebra. This already beats `Full` on residual at lower stretch (§3.7).

Tier 2: `Full` / `Level` / `Vertical` from the same solve with the measured per-mode damping. Structural, but the mode *names* are claimed (855 c14), so keep the semantics.

Tier 3: `Auto`. Requires the perceptual objective and re-exposes 460/855. Ship it last, and describe it in perceptual terms only when you have to.

Two implementation rules the forensics changed: keep `K₁ ≠ K₀` (shared K is ~10.6% off on real photos, not 32% — that is a synthetic figure), and key detection thresholds off the **long image side** with an assumed `f ≈ max(w,h)` during clustering (§3.5e) — not off focal. Add the ≥2-VP and 1.5× margin gates (§3.5g); they are cheap and they are where abstention comes from.

Original paper-derived sketch, still valid for Tier 3:

```
lens-corrected rendered raster
  → downscale (long edge ~1024) + gamma-2.2 luma        [BINARY: upright_img initial resize, Upright-MakeGrayGamma22]
  → line detection (OpenCV LSD, Apache-2.0)             [replaces AGPL ashift_lsd.c]
  → focal-length-scaled curvature filter                [steal Hugin's idea, GPL-clean reimpl]
  → per-segment line dual + ≥3 votes + ≥2 VPs + 1.5× margin  [§3.5g]
  → joint estimate of K, R, M                           [paper §4]
  → minimise E_eye + E_pic + E_img + E_f + Perkins barrier  [paper §3 — THE UNIQUE PART]
      λ_v, λ_h adaptive in rotation (σ_v as §3.5f, σ_h=π/15)
      Powell + log-barrier for the inequality constraints
  → extract rotation + focal → H = s·K₁·R·K₀⁻¹  (NOT K·R·K⁻¹; §3.1)
  → ≥2-VP and 1.5× margin gates → abstain if weak (§3.5g)
  → optional: outpaint the gaps (Lightroom has no equivalent)
```

Do **not** reimplement the geometry — implement the paper's objective and differentiate in the periphery. Expose `Transform{rotate, vertical, horizontal, x, y, scale, aspect}` so Lightroom XMP round-trips.

`.` contains a working numpy/scipy prototype of stages 2–7 (`upright_core.py`, `demo_synthetic.py`). On a synthetic 3-family building corner (camera pitched 22°, yawed, rolled):

```
  panel                        vert_tilt  horiz_tilt  p99_stretch  crop_keep
  input                           3.392°      4.176°       1.000     100.0%
  naive full rectification        0.224°      0.273°       1.305      98.0%
  energy: auto  (E=0.515)         0.234°      0.225°       1.274      97.5%
  energy: full  (E=0.410)         0.235°      0.224°       1.271      97.5%
  energy: level  (E=0.515)        0.234°      0.225°       1.274      97.5%
  energy: vertical  (E=0.515)     0.233°      0.225°       1.275      97.5%
```

Vertical convergence is cut 3.39° → 0.23°, and the perceptual energy reaches *lower* residual tilt at *lower* corner stretch than naive full rectification — while keeping 97.5 % of the frame. It also reproduces the honest caveat above: on clean input the four modes land within 0.01° and 0.004 stretch of each other, because every term is simultaneously satisfiable. The modes only diverge once noise, outliers or non-Manhattan content make the terms conflict.

---

## 9. References

- Lee, Shechtman, Wang, Lee. *Automatic Upright Adjustment of Photographs.* CVPR 2012. the CVPR 2012 paper
- Patent family — see §5 for Google Patents links.
- Adobe, *Upright: automatic perspective correction* and *Guided Upright* (Lightroom Classic Help).
- ExifTool `XMP.pm` (de-facto spec for the `crs:Perspective*` / `Upright*` keys).
- `handoff/auto-upright-survey.md`, `LIGHTROOM-UPRIGHT-FORENSICS.md`, `patent-landscape-upright.md`.
- **Reference implementation (MIT):** `../port/upright_port.py`, 102 tests green. Its detector and objective were tightened against this report; the review defects and the rounds that closed them are in §3.9. Read `../evidence/verify_metrics.txt` before quoting the accuracy numbers and `../evidence/port_fuzz.txt` before trusting `ok=True` (F1/F3 and the 45° crash are fixed, F2 partly, F4–F13 open); the review and its limits are in `port_review.txt`, the round limits in §3.9, and the stale-doc list in `doc_audit.txt`. Known gaps in `../evidence/port_status.txt`.
- **Port rounds, one file each, in order:** `../evidence/round1.txt` (VP family selection), `round2.txt` (source focal, held-out set), `round3.txt` (stale post-reassignment VP), `round4.txt` (Adobe-output scoring), `round5.txt` (Auto over-correction, failed), `round6.txt` (raw vs shape, framing dominance), `round7.txt` (output framing), `round8.txt` (destination scale), `round9.txt` (frozen scale holdout, real-photo identity failure, 45° crash), `round10.txt` (resolution-dependent gate, 80/900 accepted), `round11.txt` (family selection identified as the bottleneck; preview and radial correction rejected), `round12.txt` (VP/Mahalanobis-style selector, zero gap closed, reverted), `round13.txt` (Auto damping measured from 12,075 real Adobe pairs; hump-shaped curve, B target met, D_roll regression retained), `auto_corpus.txt` (the 4,966-photo corpus curve, controls and sensitivity).
- **Independent checks and audits:** `../evidence/verify_metrics.txt` (98/2 policy-vs-solver split, in-sample caveat, seed and perturbation confirmation, 1e-5 scale correction), `real_photos.txt` (real-photo verdict and naive/identity controls), `port_families.txt` (non-Manhattan family and extreme-pose stress, gate census), `port_fuzz.txt` (F1 horizon defect, reproducers, 34/82 unusable `ok=True`), `doc_audit.txt` (47 defects; README's framing, test-count and provenance items are since fixed, the rest stand — notably `OPERATIONS.md`'s stale `.bak` restore, its never-written TSV header, and the frozen `port_status.txt`/`port_review.txt` that still call the fixed seed-4 failure the top open defect).
- **Evidence index:** `../evidence/` — every `[BINARY]`/`[MEASURED]` claim above cites a file there. `report_audit.txt` and `site_audit.txt` record the defects found against earlier drafts of this document and of the site.
- **Reproducibility caveat:** the corpus is a live catalog, so counts drift between runs; figures are dated where they matter.

---
