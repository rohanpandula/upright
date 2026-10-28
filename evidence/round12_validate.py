"""Round-12 candidate checks; retains failures rather than asserting them away."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import os
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'port'),str(ROOT/'evidence'),str(ROOT)]
if os.environ.get('ROUND12_PORT'):
    spec=importlib.util.spec_from_file_location('upright_port',ROOT/os.environ['ROUND12_PORT'])
    up=importlib.util.module_from_spec(spec);sys.modules['upright_port']=up;spec.loader.exec_module(up)
else:
    import upright_port as up
import benchmark_accuracy as bench
import benchmark_quality as quality
from round9_probe import holdout
from benchmark_adobe import agreement
from validate_vs_adobe import CASES, SLOTS, parse_adobe


def sweep():
    rows=[]
    for scene,pose in bench.CASES.items():
        im=cv2.imread(str(ROOT/f'lrtest/in/{scene}.tif'),0);h,w=im.shape
        seg=up.detect_segments(im);truth=bench.ground_truth(pose,1600,w,h,9,-5,20)
        for seed in range(40):
            for mode in ('auto','full','level','vertical'):
                out=up.solve_ex(seg,w,h,1600,mode,seed=seed,framing=True,scaling=True)
                rows.append(dict(scene=scene,mode=mode,seed=seed,ok=out['ok'],reason=out['reason'],
                                 axis=bench.deviation(out['H'],truth,np.array([0,0,1,1]),w,h),H=out['H'].tolist()))
        print('40 seeds',scene,flush=True)
    return rows


def adobe_measure():
    cached=json.loads((ROOT/'evidence/round7.json').read_text());rows=[]
    for scene,case in CASES.items():
        image=cv2.imread(str(ROOT/f'lrtest/in/{scene}.tif'),0);h,w=image.shape
        segments=up.detect_segments(image)
        adobe=parse_adobe(ROOT/'adobe_truth'/f"img_{case['text']}.lua")
        for r in cached['rows']:
            if r['scene']!=scene:continue
            f=1600. if r['prior']=='scene_focal' else next(a['stored_focal_px'] for a in cached['native'] if a['scene']==scene)
            out=up.solve_ex(segments,w,h,f,r['mode'],seed=r['seed'],framing=True,scaling=True)
            rows.append(dict(scene=scene,prior=r['prior'],mode=r['mode'],seed=r['seed'],ok=out['ok'],reason=out['reason'],
                             H=out['H'].tolist(),final=agreement(adobe[SLOTS[r['mode']]],out['H'],w,h)))
        print('Adobe paired',scene,flush=True)
    return rows


def scope():
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    a=functions(ROOT/'evidence/round12_before_port.py');b=functions(Path(up.__file__))
    changed=[n for n in a if a[n]!=b[n]];added=sorted(b.keys()-a.keys())
    assert changed==['estimate_vanishing_points','_classify','_base_solve'],changed
    assert added==['_manhattan_margin'],added
    for n in ['detect_segments','_detect_gradient_segments','_alignment_quality','_refine_vp','_source_calibration','_energy_residuals','mode_damping','frame_output','scale_output']:
        assert a[n]==b[n],n
    print('Scope verified',changed,added,flush=True)
    return dict(changed=changed,added=added)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['sweep','heldout','quality','adobe','scope']);ap.add_argument('--label',default='candidate');a=ap.parse_args()
    result={'sweep':sweep,'heldout':holdout,'quality':lambda:quality.run(range(50)),
            'adobe':adobe_measure,'scope':scope}[a.stage]()
    (ROOT/f'evidence/round12_{a.label}_{a.stage}.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
