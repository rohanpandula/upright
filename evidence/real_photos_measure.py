"""Real-photograph evaluation of the upright port. No production changes.

Scores the shipped port (gate active) and two diagnostics (gate bypassed, naive
single-camera full rectification) against Adobe's cached transforms on a frozen
300-photo sample. Run from research/upright:

    python3.12 evidence/real_photos_measure.py gated
    python3.12 evidence/real_photos_measure.py abstain
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# A sibling round rewrites the port while this evaluation runs. Set
# RP_PORT_DIR to a frozen byte-identical copy to avoid a mixed snapshot.
FROZEN = Path(os.environ.get('RP_PORT_DIR', ROOT / 'port'))
sys.path[:0] = [str(FROZEN), str(ROOT / 'port'), str(ROOT / 'evidence'), str(ROOT)]
import upright_port as up
from benchmark_adobe import agreement
from round9_probe import load_photo, orient, json_default

POLICY_SHA = '60f842ea1d03356f4c411b00f3e0e0802dc0a0a8f20a86a468268979fa7b14ad'
# The gate was rewritten by a sibling round while this evaluation ran; both
# policies are scored. Pass the expected digest as argv[3].

C = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])


def safe_frame_scale(H):
    """Framing then scaling; None if either raises."""
    try:
        return up.scale_output(up.frame_output(H))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None


def quad_area(H):
    q = C @ np.asarray(H).T
    if np.min(q[:, 2]) <= 0 or not np.isfinite(q).all():
        return None
    q = q[:, :2] / q[:, 2:]
    return float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - q[:, 1] * np.roll(q[:, 0], -1)) / 2)


def measure(r):
    start = time.time()
    cv2.setNumThreads(1)
    try:
        gray, Q, decode = load_photo(r)
    except Exception as exc:
        return dict(image_id=r['image_id'], path=r['path'], decode_error=repr(exc))
    wh, ww = gray.shape
    w, h = int(r['width']), int(r['height'])
    if r['orientation'] in ('BC', 'DA', 'AD', 'CB'):
        w, h = h, w
    f = up.metadata_focal_px(r['meta'].get('UprightFocalLength35mm'), w, h)
    seg = up.detect_segments(gray) * [w / ww, h / wh, w / ww, h / wh]
    seg_n = seg / np.array([w, h, w, h], dtype=float)

    out = dict(image_id=r['image_id'], path=r['path'], width=w, height=h,
               aspect=max(w, h) / min(w, h), portrait=h > w,
               orientation=r['orientation'], segments=len(seg),
               input_focal_mm=r['meta'].get('UprightFocalLength35mm'),
               decode=decode, rows=[])
    for mode, slot in (('auto', '1'), ('full', '2')):
        native = np.array(r['transform'][slot]).reshape(3, 3)
        shipped = up.solve_ex(seg, w, h, f, mode, seed=0, framing=True, scaling=True)
        bare = up.solve_ex(seg, w, h, f, mode, seed=0)
        got = up._base_solve(seg_n, w, h, f, up._FULL_WEIGHTS, mode, 0)
        naive = up._base_solve(seg_n, w, h, f, up._FULL_WEIGHTS, 'full', 0,
                               shared_k=True)
        variants = {'shipped': shipped['H'], 'shipped_bare': bare['H']}
        info = None
        if got is not None:
            info = got[1]
            ungated = got[0]
            variants['ungated'] = ungated
            fs = safe_frame_scale(ungated)
            variants['ungated_fs'] = fs if fs is not None else ungated
            variants['ungated_fs_failed'] = fs is None
        else:
            variants['ungated'] = np.eye(3)
            variants['ungated_fs'] = np.eye(3)
            variants['ungated_fs_failed'] = True
        if naive is not None:
            variants['naive'] = naive[0]
            fs = safe_frame_scale(naive[0])
            variants['naive_fs'] = fs if fs is not None else naive[0]
        else:
            variants['naive'] = np.eye(3)
            variants['naive_fs'] = np.eye(3)

        def score(H):
            # Catalog maps are sensor-axis; compare in that gauge.
            try:
                m = agreement(native, np.linalg.inv(Q) @ H @ Q, w, h)
            except ValueError as exc:
                return dict(error=str(exc), raw_rms_pct=None)
            m['area'] = quad_area(H)
            return m

        metrics = {k: score(v) for k, v in variants.items() if k != 'ungated_fs_failed'}
        row = dict(mode=mode, ok=shipped['ok'], reason=shipped['reason'],
                   quality=shipped['quality'], metrics=metrics,
                   native_perspective=float(np.linalg.norm(native[2, :2])),
                   native_identity_pct=agreement(native, np.eye(3), w, h)['raw_rms_pct'],
                   ungated_fs_failed=variants['ungated_fs_failed'],
                   rotation_deg=np.degrees(info['rotation']).tolist() if info else None,
                   damping=info['damping'] if info else None,
                   keystone_deg=float(np.degrees(info['vertical_keystone_rad'])) if info else None,
                   source_focal_mm=up.focal_35mm_from_px(info['source_focal_px'], w, h) if info else None,
                   calibration=info['calibration'] if info else None)
        out['rows'].append(row)
    out['seconds'] = time.time() - start
    return out


def run(sample_name, out_name, workers):
    sample = json.loads((ROOT / 'evidence' / sample_name).read_text())
    path = ROOT / 'evidence' / out_name
    with path.open('w') as fh:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(measure, sample), 1):
                fh.write(json.dumps(result, default=json_default) + '\n')
                fh.flush()
                if i % 10 == 0:
                    print(i, flush=True)


if __name__ == '__main__':
    stage = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    expected = sys.argv[3] if len(sys.argv) > 3 else POLICY_SHA
    actual = hashlib.sha256((FROZEN / 'upright_port.py').read_bytes()).hexdigest()
    assert actual == expected, (actual, expected)
    tag = '' if expected == POLICY_SHA else '_new'
    if stage == 'gated':
        run('round9_sample.json', f'real_photos_raw{tag}.jsonl', workers)
    elif stage == 'abstain':
        run('real_photos_abstain_sample.json', f'real_photos_abstain{tag}.jsonl', workers)
    else:
        raise SystemExit('stage must be gated or abstain')
