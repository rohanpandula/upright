"""Probe robust final VP fits in centered isotropic coordinates."""
import sys,importlib.util
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'port'),str(Path(__file__).resolve().parents[1])]
from unittest.mock import patch
import numpy as np, cv2
import upright_port as up,benchmark_accuracy as b,make_heldout_images as ho
spec=importlib.util.spec_from_file_location('original',str(Path(__file__).with_name('round3_original_port.py')))
orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)


def refit_vp(s,w,h,v):
 d=max(w,h)
 N=np.array([[1/d,0,-w/2/d],[0,1/d,-h/2/d],[0,0,1.]])
 sn=(s-[w/2,h/2,w/2,h/2])/d
 L=up._homogeneous_lines(sn);L/=np.linalg.norm(L[:,:2],axis=1)[:,None]
 mids=np.column_stack([(sn[:,:2]+sn[:,2:])/2,np.ones(len(s))])
 lengths=np.linalg.norm(s[:,2:]-s[:,:2],axis=1)
 v=up._normalize(N@v)
 for i in range(20):
  denom=np.maximum(np.linalg.norm(v[:2]-mids[:,:2]*v[2],axis=1),1e-8)
  residual=L@v/denom
  weight=np.sqrt(lengths/(denom**2*(1+(residual/(4/(d-1)))**2)))
  _,_,vt=np.linalg.svd(L*weight[:,None]);v=up._normalize(vt[-1])
 return up._normalize(np.linalg.inv(N)@v)


def robust(s,w,h,seed=0):
 vps=orig.estimate_vanishing_points(s,w,h,seed)
 for v in vps:v['vp']=refit_vp(s[v['inliers']],w,h,v['vp'])
 return vps

if __name__=='__main__':
 inputs=[]
 for n in b.CASES:
  im=cv2.imread(str(b.ROOT/'lrtest/in'/(n+'.tif')),0);h,w=im.shape
  inputs.append((n,im,1600,b.ground_truth(b.CASES[n],1600,w,h,9,-5,20),np.array([0,0,1,1])))
 c=ho.CASES[8];im,t,k,_=ho.render(c);inputs.append(('H09',im,ho.camera(c)[1][0,0],t,k))
 for n,im,f,t,k in inputs:
  h,w=im.shape;s=up.detect_segments(im)
  for seed in range(10):
   with patch.object(up,'estimate_vanishing_points',robust):o=up.solve_ex(s,w,h,f,'full',seed=seed)
   K=up._intrinsics_norm(o['source_focal_px'],w,h);H=K@up._rotation(*o['rotation'])@np.linalg.inv(K)
   print(n,seed,o['ok'],b.deviation(H,t,k,w,h),o['source_focal_px'],flush=True)
