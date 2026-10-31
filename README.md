<div align="center">
  <img src="icon.svg" width="96" alt="Upright logo: a skewed frame straightened">

  # upright
  *Reverse-engineered Lightroom Upright, and an MIT port of what it does*

  [![test](https://img.shields.io/github/actions/workflow/status/rohanpandula/upright/test.yml?style=flat-square&label=test)](https://github.com/rohanpandula/upright/actions)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-3c873a?style=flat-square)](https://python.org)
  [![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

  [What it does](#what-it-does) • [Status](#status) • [Install](#install) • [Usage](#usage) • [Repository](#repository) • [Patents](#patents)
</div>

Adobe Lightroom Classic ships a perspective correction called Upright. It is not a black box:
it ships a specific piece of algebra and a specific behaviour, and both can be measured from
outside the program. The algebra is recoverable from the transforms Lightroom writes into its own
catalogue. The behaviour is recoverable by driving the real application on scenes where the camera
rotation is known exactly. This repository does both, and publishes a reference implementation.

## What it does

Upright ships `H = s · K₁ · R · K₀⁻¹`. That is a camera rotation re-projected through a *second,
different* camera, rather than the shared-intrinsics form `K·R·K⁻¹` that almost everyone reaches
for first.

The difference is not cosmetic. Fitting Adobe's own stored transforms, the shared-`K` form stalls
at a leftover error of `3.19e-01` while the two-camera form reaches `5.65e-16`, which is machine
precision. The two independent intrinsics are what let Adobe make a wall parallel without forcing
the perspective recovery a physically correct rotation would demand.

The second finding is behavioural. `Auto` deliberately under-corrects, and how far it stops short
is a smooth function of camera pitch, gated on whether the verticals need fixing at all. On a
frame with parallel verticals and converging horizontals it does nothing, moving horizontal
convergence by 0.01° where `Full` removes 96%.

- **Model recovered** from 12,883 harvested transforms, with an orthogonality test that separates
  the two-camera form from every single-focal alternative
- **Damping curve measured** over 29 controlled solves and a 4,966-photo corpus, including the
  `π/12` and `π/15` literals located in the shipping binary
- **Guided mode cracked**, including its storage format and the digest that gates re-analysis
- **Port shipped** as MIT Python, with every constant tagged `PROVEN`, `MEASURED` or `CHOICE`

## Status

> [!WARNING]
> The port works on synthetic scenes and is poor on real photographs. It returns identity on about
> 90% of real frames and loses to doing nothing on many of the rest. Use it to understand the
> algorithm, not yet to correct your photos.

| | |
|---|---|
| Model recovery | Solid. Fits at machine precision, disagrees with every single-focal form |
| Behaviour measurement | Solid. Curves fitted against real photos, not the paper |
| Port on synthetic scenes | Good. 0.29–1.96% error against Adobe with framing and scaling on |
| Port on real photographs | **Poor.** 40.7% loss rate against plain identity |

The open defect is family selection: choosing which detected lines belong to which direction. With
oracle selection the `Full` median error falls from 3.66% to 0.93%, and the current selector
captures none of that. [CONTRIBUTING.md](CONTRIBUTING.md) covers it in detail.

## Install

```bash
python3 -m pip install numpy 'opencv-python-headless<5'
```

The port needs only numpy and OpenCV, and CI enforces that by running the port's own tests with
nothing else installed. Three of the test files also reach the older prototype in
`upright_core.py`, which uses scipy and Pillow, so the full suite needs those too:

```bash
python3 -m pip install numpy 'opencv-python-headless<5' scipy pillow pytest
```

## Usage

```python
import cv2, upright_port as up

image = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
h, w = image.shape

segments = up.detect_segments(image)
result = up.solve_ex(
    segments, w, h,
    focal_px=1600.0,
    mode="auto",
    framing=True,
    scaling=True,
)

if result["ok"]:
    warp = result["H"]  # 3x3, normalised frame coordinates
```

`mode` accepts `auto`, `full`, `level`, `vertical` or `guided`. Guided takes exactly four segments
as `"x1,y1,x2,y2"` strings and is the strongest mode, because it does not depend on the detector.

> [!IMPORTANT]
> Framing and scaling are opt-in and off by default. They are not cosmetic: the same solve scores
> about 33% against Adobe without them and 0.587% with them. Leaving them off gives you a centred,
> unframed warp.

Every result carries `ok` and `reason`. Check `ok`. An identity with `ok=False` means the solver
declined, which is usually the right call on a frame with no usable structure.

> [!NOTE]
> `benchmark_adobe.py` calls `solve_ex` bare, so it reports the unframed ~31% path. The 0.587%
> figure is `framing=True, scaling=True`. Never compare the two.

## Repository

| path | |
|---|---|
| `port/` | the implementation, its tests and benchmarks |
| `docs/REPORT.md` | the full write-up, with provenance on every claim |
| `docs/index.html` | the same findings as a [browsable page](https://rohanpandula.github.io/upright/) |
| `evidence/` | the measurements each claim rests on, one file per round |
| `lrplugin/` | the Lightroom plugin that drives the real application |
| `analysis/` | fitting and corpus scripts |
| `heldout/`, `lrtest/in/`, `adobe_truth/` | fixtures the tests read |

Binary slices of `CameraRaw` and the large per-attempt dumps are gitignored. The scripts that
produce them are in `evidence/`.

## Tests

```bash
cd port && python3 -m pytest -q
```

102 tests, covering the focal conversion, round-trip recovery, the orthogonality gap, guided
convergence, the horizon and degenerate-input cases, and the recovered framing and scaling rules.

## Patents

MIT covers the copyright in this code and grants no patent rights. Adobe holds a family on this
behaviour (US 9,008,460, 9,582,855, 9,098,885, 9,519,954, 9,727,787), expiring from 2033.

The adaptive weights are unclaimed across all 100 claims, so the numbers can be changed freely.
`Auto` is the exposed mode, because the broadest claim covers a transform chosen "according to
human perception-based criteria". `Guided` clears four of the five independent claims and is the
safer mode to build a product on.

Nothing here is legal advice.

---

No Adobe source was used. The port is clean-room; the binary was read only for constants and
control flow. Licensed under [MIT](LICENSE).
