"""Perturbation robustness of the held-out accuracy claim. MIT."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'port'))
sys.path.insert(0, str(ROOT / 'evidence'))
sys.path.insert(0, str(ROOT))
import upright_port as up  # noqa: E402
import benchmark_accuracy as bench  # noqa: E402
import make_heldout_images as heldout  # noqa: E402

SEEDS = range(5)


def solve_score(image, truth, kind, w, h, focal):
    seg = up.detect_segments(image)
    vals, rej = [], 0
    for seed in SEEDS:
        out = up.solve_ex(seg, w, h, focal, 'full', seed=seed)
        if not out['ok']:
            rej += 1
        vals.append(bench.deviation(out['H'], truth, kind, w, h))
    return np.array(vals), rej, len(seg)


def variants(image):
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    yield 'jpeg75', cv2.imdecode(buf, 0)
    yield 'blur0.8', cv2.GaussianBlur(image, (0, 0), 0.8)
    yield 'contrast0.85', np.clip(128 + 0.85 * (image.astype(float) - 128), 0, 255).astype(np.uint8)
    yield 'contrast1.20', np.clip(128 + 1.20 * (image.astype(float) - 128), 0, 255).astype(np.uint8)
    for f in (0.75, 1.5):
        small = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        yield f'resize{f}', small


def main():
    base = {}
    results = {}
    for case in heldout.CASES:
        name = case['name']
        image = cv2.imread(str(ROOT / 'heldout' / (name + '.png')), 0)
        z = np.load(ROOT / 'heldout' / (name + '.npz'))
        truth, kind = z['segments'], z['kind']
        w, h = case['w'], case['h']
        focal = float(heldout.camera(case)[1][0, 0])
        b, rej, n = solve_score(image, truth, kind, w, h, focal)
        base[name] = (b, rej, n)
        for tag, im in variants(image):
            if tag.startswith('resize'):
                f = 0.75 if tag == 'resize0.75' else 1.5
                nw, nh = int(round(w * f)), int(round(h * f))
                im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
                vals, r, ns = solve_score(im, truth * f, kind, nw, nh, focal * f)
            else:
                vals, r, ns = solve_score(im, truth, kind, w, h, focal)
            results.setdefault(tag, []).append((name, vals, r, ns))

    def summarize(tag, rows):
        allv = np.concatenate([r[1].ravel() for r in rows])
        print(f'{tag:14} mean {allv.mean():.6f} worst {allv.max():.6f} '
              f'over0.1 {int((allv >= 0.1).sum())}/{allv.size} rejects {sum(r[2] for r in rows)} '
              f'minseg {min(r[3] for r in rows)}')
        for name, vals, rej, ns in rows:
            if vals.max() >= 0.1 or rej:
                print(f'   {name} worst {vals.max():.6f} rejects {rej} segs {ns}')

    print('variant        statistic (degrees, all scenes x seeds 0..4)')
    summarize('baseline', [(n, v[0], v[1], v[2]) for n, v in base.items()])
    for tag, rows in results.items():
        summarize(tag, rows)
    print('\nper-scene worst:')
    for tag, rows in results.items():
        print(tag, ' '.join(f'{n}:{v.max():.4f}' for n, v, _, _ in rows))


if __name__ == '__main__':
    main()
