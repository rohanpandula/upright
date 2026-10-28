# Empirical validation harness — driving real Lightroom Classic

These scripts make the *actual* Lightroom Classic on this machine run its own
Upright solver, then compare the result against our reimplementation and against
known ground truth. Nothing here is inferred from documentation.

## Pipeline

1. `make_test_images.py` — generates 1400×1000 synthetic Manhattan scenes with
   exactly known pitch/yaw/roll at a known true focal (1600 px). The images are
   deliberately trivial for a line detector: this isolates *what Upright does
   once it has the lines*, which is the part being reverse-engineered.

2. Import them into a catalog, then select the folder containing them in
   Lightroom's **Library → Folders** panel. This matters: the Develop module
   only renders a photo that is inside the current filmstrip source, and
   `catalog:setSelectedPhotos` silently no-ops otherwise. `applyDevelopSettings`
   likewise does nothing while no photo is displayed.

3. Make Lightroom click its own **Upright → Auto** button. The Upright solver
   computes *all six* mode slots in one pass, so a single click populates
   `UprightTransform_0..5`.

4. Read the result straight out of the catalog:
   `Adobe_imageDevelopSettings.text` holds
   `UprightTransform_N = "a,b,c,..."` (row-major 3×3, normalised [0,1] frame
   coordinates) plus `UprightFocalLength35mm`, `UprightCenterNormX/Y`,
   `UprightFocalMode`, `UprightCenterMode`, `UprightDependentDigest`.

## Analysis

| script | what it establishes |
|---|---|
| `test_model_exact.py` | **The shipped algebra is exactly `H = s·K₁·R·K₀⁻¹`.** Minimises the orthogonality defect `‖N Nᵀ − s²I‖` for `N = K₁⁻¹HK₀`. Reaches `5.7e-16`; the naive shared-`K` model stalls at `3.2e-01`. |
| `test_model_orthogonality.py` | Same test restricted to subsets of the intrinsics, showing *which* freedoms are needed (`f₀` alone is not enough; the destination principal point is not enough). |
| `validate_vs_adobe.py` | Detector-free ground-truth comparison: residual tilt, vertical/horizontal convergence, stretch and crop for Adobe vs our prototype, per mode. |
| `fit_adobe_model.py`, `fit_adobe_constrained.py` | Earlier, looser optimiser fits; kept for the record. Superseded by the orthogonality test, which needs no initial guess. |

Raw evidence is in `evidence/`; the harvested Adobe transforms are in
`adobe_truth/`.

## Driving Lightroom from the shell

There is no usable CLI and the LR SDK cannot trigger the solver, so the harness
drives the GUI:

- `lrplugin/uprightprobe.lrdevplugin/` — a Lightroom plugin exposing a
  `Library → Plug-in Extras` menu item. Note `Init.lua` is **never executed** by
  Lightroom Classic; only the menu action runs. Every `pcall` must be
  `LrTasks.pcall` or Lua 5.1 refuses to yield.
- `/tmp/click.js`, `/tmp/scroll.js`, `/tmp/ocrall.js` — JXA helpers posting
  `CGEvent`s and running Vision OCR, used to locate and click buttons by their
  on-screen text. Coordinates are in **points**: the display is 2560×1080 pt
  (5120×2160 px, 2× scale), so screen pixels from a screenshot must be halved.
- Lightroom's window is remembered at 900×930 pt. If it is wider than the screen
  the entire right-hand panel (including Transform) lives off-screen and silently
  OCRs as blank — `Window → Fill` fixes it.

## Two traps worth remembering

- **"No photo selected."** Means the photo is outside the current Library
  filter/source, not that selection failed. Fix the *source*, not the selection.
- Clicking **Off** then **Auto** is what forces a re-solve; clicking Auto when it
  is already active is a no-op. Confirming the solve really re-ran is done by
  watching `UprightFocalLength35mm` change (e.g. 38.12 → 43.49 on a real photo).
