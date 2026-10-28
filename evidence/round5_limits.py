"""Damping/framing counterfactuals. Adobe H is used only by offline scoring."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'port')); sys.path.insert(0,str(ROOT))
import upright_port as up
from benchmark_adobe import agreement,mapped_points,visible_verticals,convergence
from validate_vs_adobe import CASES
from nm_analyze import sim_residual
import make_test_images as M
import cv2

grid=np.array([[(u,v) for u in np.linspace(60,M.W-60,9)] for v in np.linspace(60,M.H-60,7)]).reshape(-1,2)
def ideal_fraction(H, pose):
    candidates=[]
    for p in (pose, [pose[0],0.,pose[2]]):
        gt=M.ground_truth_H(*p)
        candidates.append((sim_residual(H,gt,grid)[0],sim_residual(np.eye(3),gt,grid)[0]))
    residual,baseline=min(candidates,key=lambda s:s[0])
    return 1-residual/baseline
def run():
 data=json.loads((ROOT/'evidence/round4_adobe.json').read_text()); rows=[]
 for row in data['rows']:
  if row['version']!='after' or row['prior']!='scene_focal' or row['mode']!='auto' or row['seed']!=0:continue
  name=row['scene'];w,h=M.W,M.H; R=np.radians(row['rotation_deg']);K0=up._intrinsics_norm(row['source_focal_px'],w,h);Ki=np.linalg.inv(K0)
  K1=np.array(row['H'])@K0@up._rotation(*(R*row['rotation_damping'])).T;K1/=K1[2,2]
  Ha=np.array(row['adobe_H']);a=mapped_points(Ha,w,h);c=CASES[name];pose=[c[k] for k in ('rx','ry','rz')]
  vs=visible_verticals(pose,1600,w,h);conv=convergence(np.eye(3),vs,w,h)
  def mapping(d):return K1@up._rotation(*(R*d))@Ki
  reconstructed=mapping(np.array(row['rotation_damping']))
  np.testing.assert_allclose(reconstructed/reconstructed[2,2],row['H'],atol=1e-12)
  def residual(d):return (mapped_points(mapping(d),w,h)-a).ravel()
  def score(d):return agreement(Ha,mapping(d),w,h)['shape_rms_pct']
  starts=[row['rotation_damping'],[0,0,0],[.5,.5,.5],[1,1,1]]
  rawfit=min((least_squares(residual,s,bounds=(0,1),max_nfev=300) for s in starts),key=lambda f:np.linalg.norm(f.fun))
  shapefit=min((minimize(score,s,bounds=[(0,1)]*3,method='Nelder-Mead',options={'maxiter':1000,'xatol':1e-10}) for s in starts),key=lambda f:f.fun)
  for tag,d in [('round4',np.array(row['rotation_damping'])),('best_raw',rawfit.x),('best_shape',shapefit.x)]:
   H=mapping(d);ag=agreement(Ha,H,w,h)
   rows.append(dict(scene=name,variant=tag,damping=list(d),**ag,framing_squared_fraction=1-(ag['shape_rms_pct']/ag['raw_rms_pct'])**2,removed=1-convergence(H,vs,w,h)/conv,ideal_fraction=ideal_fraction(H,pose)))
  # Force the SAME visible-convergence fraction as Adobe, without fitting any framing.
  sy,sz=row['rotation_damping'][1:];target=convergence(Ha,vs,w,h)
  lo,hi=0.,1.
  for _ in range(40):
   mid=(lo+hi)/2
   if convergence(mapping(np.array([mid,sy,sz])),vs,w,h)>target:lo=mid
   else:hi=mid
  d=np.array([(lo+hi)/2,sy,sz]);ag=agreement(Ha,mapping(d),w,h)
  assert abs(convergence(mapping(d),vs,w,h)-target)<1e-8
  rows.append(dict(scene=name,variant='oracle_native_convergence_only',damping=list(d),**ag,ideal_fraction=ideal_fraction(mapping(d),pose)))
  # Counterfactual only: a generic source-centre anchor, no cache-derived transform.
  H=mapping(np.array(row['rotation_damping']));q=H@np.array([.5,.5,1]);q=q[:2]/q[2]
  T=np.eye(3);T[:2,2]=.5-q
  ag=agreement(Ha,T@H,w,h)
  rows.append(dict(scene=name,variant='diagnostic_center_anchor',**ag,ideal_fraction=ideal_fraction(T@H,pose)))
 mild=[]
 for name,path in [('NM_single_wall','nm'),('NM_railing','nm'),('NM_curves','nm'),('NM_three_plane','nm'),('CT_manhattan2','nm2')]:
  image=cv2.imread(str(ROOT/'lrtest'/path/(name+'.tif')),0);h,w=image.shape
  seg=up.detect_segments(image)
  for mode in ('auto','full'):
   out=up.solve_ex(seg,w,h,1600,mode)
   mild.append(dict(scene=name,mode=mode,ok=out['ok'],reason=out['reason'],rotation_deg=np.degrees(out['rotation']).tolist(),damping=out['damping'],keystone_deg=np.degrees(out['vertical_keystone_rad']),ideal_fraction=ideal_fraction(out['H'],[M.PITCH15,M.YAW6,0.])))
   print(mild[-1],flush=True)
 return dict(counterfactuals=rows,mild=mild)
if __name__=='__main__':
 result=run();(ROOT/'evidence/round5_limits.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in result['counterfactuals']:print(r,flush=True)
