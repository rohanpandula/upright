"""Targeted causal interventions; no world truth enters solver interventions."""
import sys
import importlib.util
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'port'),str(Path(__file__).resolve().parents[1])]
import numpy as np
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('round3_original',
    Path(__file__).with_name('round3_original_port.py'))
up=importlib.util.module_from_spec(spec)
spec.loader.exec_module(up)
import benchmark_accuracy as b
import make_heldout_images as ho

original=up.estimate_vanishing_points

def refit(segments,w,h,seed=0):
    vps=original(segments,w,h,seed)
    lines=up._homogeneous_lines(segments)
    lines/=np.linalg.norm(lines[:,:2],axis=1)[:,None]
    for vp in vps:
        _,_,vt=np.linalg.svd(lines[vp['inliers']])
        vp['vp']=up._normalize(vt[-1])
    return vps

if __name__=='__main__':
 c=ho.CASES[8];im,t,k,_=ho.render(c);w,h=c['w'],c['h'];s=up.detect_segments(im);f=ho.camera(c)[1][0,0]
 for seed in range(5):
  for name in ('original','no_horizon','refit_final_supports','refit_no_horizon'):
   weights=dict(up._MODE_WEIGHTS['full'])
   if 'no_horizon' in name: weights['w_eye']=0
   with patch.dict(up._MODE_WEIGHTS,full=weights), patch.object(up,'estimate_vanishing_points',refit if 'refit' in name else original):
    o=up.solve_ex(s,w,h,f,'full',seed=seed)
   print(seed,name,b.deviation(o['H'],t,k,w,h),o['quality'],flush=True)
