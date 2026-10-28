"""Seed sweep plus the full accuracy benchmark; preserves historical evidence.

Run before/after changes with --output evidence/round1_before.json (or after).
Python 3.12; detections are cached once per fixture, seeds affect VP search only.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import benchmark_accuracy as bench
import upright_port as up


def family_quality(H, segments, vps, w, h, focal):
    seg_n = segments / [w, h, w, h]
    vertical, horizontal = up._classify(vps, seg_n)
    K0i = np.linalg.inv(up._intrinsics_norm(focal, w, h))
    rows = []
    for family in [vertical, *horizontal]:
        if family is None:
            continue
        s = seg_n[family['inliers']]
        lines = up._homogeneous_lines(s) @ np.linalg.inv(H)
        dx, dy = np.abs(lines[:, 1] * w), np.abs(lines[:, 0] * h)
        angles = np.degrees(np.arctan2(dx, dy) if family is vertical else np.arctan2(dy, dx))
        lengths = np.hypot((s[:,2]-s[:,0])*w, (s[:,3]-s[:,1])*h)
        order = np.argsort(angles)
        median = angles[order[np.searchsorted(np.cumsum(lengths[order]), lengths.sum()/2)]]
        ray = up._normalize(K0i @ family['vp'])
        p = family['vp']
        rows.append(dict(vertical=family is vertical, vp=p.tolist(), count=len(s),
                         weight=family['weight'], origin_lateral=abs(p[0])/np.hypot(*p[:2]),
                         camera_lateral=abs(ray[0]), median_deg=float(median)))
    return rows


def run():
    rows = {}
    saved = {}
    for name, pose in bench.CASES.items():
        image = cv2.imread(str(bench.ROOT / 'lrtest/in' / (name+'.tif')), 0)
        h,w = image.shape
        seg = up.detect_segments(image)
        saved[name] = seg
        truth = bench.ground_truth(pose,1600,w,h,9,-5,20)
        rows[name] = []
        for seed in range(50 if name == 'D_roll' else 10):
            out = up.solve_ex(seg,w,h,1600,'full',seed=seed)
            rows[name].append(dict(seed=seed, ok=out['ok'], reason=out['reason'],
                                  axis=bench.deviation(out['H'],truth,np.array([0,0,1,1]),w,h),
                                  quality=out.get('quality'),
                                  families=family_quality(out['H'],seg,out['vps'],w,h,1600)))
        print(name, 'worst',max(rows[name],key=lambda r:max(r['axis']))['axis'], flush=True)
    return dict(protocol='Python 3.12; Full; true focal; frozen rasters; independent wall-A world lines; degrees',
                seeds=rows, accuracy=bench.run()), saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--segments', type=Path)
    args = parser.parse_args()
    output, segments = run()
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    if args.segments:
        np.savez_compressed(args.segments,**segments)
