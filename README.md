# Upright

A reverse-engineered account of Adobe Lightroom Classic's Upright and keystone correction, plus an
MIT-licensed Python port of what was recovered.

Lightroom's Upright is not a black box. It ships a specific piece of algebra and a specific
behaviour, and both are measurable from the outside: the algebra from the transforms Lightroom
writes into its own catalogue, the behaviour by driving the real application on scenes where the
camera rotation is known exactly. That is what this repository contains.

**The short version of what was found:** Upright ships `H = s · K₁ · R · K₀⁻¹`, a rotation
re-projected through a second, different camera rather than the shared-intrinsics form almost
everyone guesses. That form fits Adobe's own stored transforms to `5.65e-16`; the shared-`K` form
stalls at `3.19e-01`. The second finding is behavioural: `Auto` deliberately under-corrects, and
the amount is a smooth function of camera pitch, keyed on a gate that does nothing at all on
already-parallel verticals.

Read [the write-up](docs/REPORT.md) or [the page](site/index.html).

## Status

Be clear about this before using anything here.

| | |
|---|---|
| Model recovery | Solid. Fits at machine precision, and disagrees with every single-focal form |
| Behaviour measurement | Solid. 12,883 transforms harvested, curves fitted against real photos |
| Port on synthetic scenes | Good. 0.29–1.96% error against Adobe with framing and scaling on |
| Port on real photographs | **Poor.** Returns identity on about 90% of frames and loses to doing nothing on many of the rest |

The port is a reference implementation of the findings, not a finished tool. The dominant
remaining defect is choosing which detected lines belong to which direction: with oracle selection
the `Full` median error falls from 3.66% to 0.93%, and the current selector captures none of that.

## Install

```
python3 -m pip install numpy opencv-python-headless
```

Only numpy and OpenCV are needed. The test suite needs `pytest`.

## Use

```python
import cv2, upright_port as up

image = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
h, w = image.shape
segments = up.detect_segments(image)
result = up.solve_ex(segments, w, h, focal_px=1600.0, mode="auto",
                     framing=True, scaling=True)
if result["ok"]:
    warp = result["H"]          # 3x3, normalised frame coordinates
```

`mode` is one of `auto`, `full`, `level`, `vertical`, `guided`. Guided takes exactly four segments
as `"x1,y1,x2,y2"` strings and is the strongest mode, because it does not need the detector.

Framing and scaling are opt-in and off by default. They matter a lot: the same solve scores ~33%
against Adobe without them and 0.587% with them. Turning them off gives you a centred, unframed
warp.

Every result carries `ok` and `reason`. Check `ok`. A returned identity with `ok=False` means the
solver declined, which is usually correct behaviour on a frame with no usable structure.

## What is here

| path | |
|---|---|
| `port/` | the implementation, tests and benchmarks |
| `docs/REPORT.md` | the full write-up, with provenance on every claim |
| `site/` | a self-contained page, open `site/index.html` |
| `evidence/` | the measurements each claim rests on |
| `lrplugin/` | the Lightroom plugin that drives the real application |
| `heldout/`, `lrtest/in/`, `adobe_truth/` | fixtures used by the tests |

`re/vp/*.bin` and the large per-round JSON dumps are gitignored. They are byte slices of the
`CameraRaw` binary and per-attempt records; the scripts that produce them are in `evidence/`.

## Tests

```
cd port && python3 -m pytest -q
```

102 tests. They cover the focal conversion, round-trip recovery, the model orthogonality gap,
guided convergence, the horizon and degenerate-input cases, and the recovered framing and scaling
rules.

## Patents

MIT covers the copyright in this code. It does not grant patent rights, and Adobe has a patent
family on this behaviour (US 9,008,460, 9,582,855, 9,098,885, 9,519,954, 9,727,787, expiring from
2033). The adaptive weights are unclaimed across all 100 claims, so the numbers can be changed
freely. `Auto` is the exposed mode, because the broadest claim covers a transform chosen
"according to human perception-based criteria". `Guided` clears four of the five independent
claims and is the safer mode to build on.

Nothing here is legal advice, and if you are shipping commercially you should get some.

## Licence

MIT. See [LICENSE](LICENSE).

No Adobe source was used. The port is a clean-room implementation of measured behaviour; the
binary was read only for constants and control flow.
