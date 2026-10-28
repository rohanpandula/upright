"""
Dense pure-pitch sweep for the vertical adaptive-weight constant sigma_v.

40 images: pitch = 2..40 deg step 2, two independent seeds each.
Seed controls only nuisance factors (sensor grain + a few px of framing
jitter); the camera rotation is EXACTLY pure pitch (ry = rz = 0), so the
ground-truth tilt is the pitch angle itself.

Writes TIFFs to lrtest/abstain (the directory the ImportSweep plugin reads).
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_test_images import render

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lrtest", "abstain")
ANGLES = range(2, 41, 2)
SEEDS = (1, 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".tif"):
            os.remove(os.path.join(OUT, f))
    rng = np.random.default_rng(20261027)
    for deg in ANGLES:
        for seed in SEEDS:
            a = render(-np.deg2rad(deg), 0.0, 0.0)
            # seed-dependent nuisance: sub-pixel framing jitter + grain
            dx, dy = rng.uniform(-4, 4, 2)
            img = Image.fromarray(a.astype(np.uint8)).transform(
                a.shape[1::-1], Image.AFFINE, (1, 0, dx, 0, 1, dy),
                resample=Image.BILINEAR, fillcolor=(238, 236, 232))
            arr = np.asarray(img, dtype=np.float64)
            g = np.random.default_rng(1000 * seed + deg)
            arr = np.clip(arr + g.normal(0, 3.5, arr.shape), 0, 255)
            name = f"V{deg:02d}S{seed}.tif"
            Image.fromarray(arr.astype(np.uint8)).save(
                os.path.join(OUT, name), compression="tiff_lzw")
    print(f"wrote {len(list(ANGLES)) * len(SEEDS)} images to {OUT}")


if __name__ == "__main__":
    main()
