"""Run candidate Auto responses through the actual solver; diagnostic, no shipped change."""
from pathlib import Path
import sys, json, math, importlib.util
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'port'));sys.path.insert(0,str(ROOT))
from benchmark_adobe import agreement, visible_verticals, convergence
from validate_vs_adobe import CASES, parse_adobe
from nm_analyze import sim_residual
from round5_limits import ideal_fraction
import make_test_images as M
spec=importlib.util.spec_from_file_location('round5_frozen',ROOT/'evidence/round5_original_port.py')
up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
grid=np.array([[(u,v) for u in np.linspace(60,M.W-60,9)] for v in np.linspace(60,M.H-60,7)]).reshape(-1,2)
original=up._auto_pitch_scale
sweep=json.loads((ROOT/'evidence/sigma_v_precise.json').read_text())['rows']
sweep_angles=sorted({r['theta'] for r in sweep})
sweep_means=[np.mean([r['frac'] for r in sweep if r['theta']==a and r.get('frac') is not None]) for a in sweep_angles]

def candidate(label):
 def scale(r,f,K1,Ki,sv,w,h,ks):
  if label=='round4':return original(r,f,K1,Ki,sv,w,h,ks)
  if label=='keystone_euler':return f[0]
  if label=='empirical_sweep_physical':
   q=float(np.interp(abs(math.degrees(r[0])),sweep_angles,sweep_means))
   return original(r,(q,f[1],f[2]),K1,Ki,sv,w,h,ks)
  sigma=math.radians(13.33 if label=='wrong_sigma_camera_euler' else 22.26)
  q=up._saturate(2.64*math.exp(-r[0]**2/(2*sigma**2)))
  if label in ('camera22_euler','wrong_sigma_camera_euler'):return q
  return original(r,(q,f[1],f[2]),K1,Ki,sv,w,h,ks)
 return scale

labels=('round4','camera22_physical','keystone_euler','camera22_euler','wrong_sigma_camera_euler','empirical_sweep_physical')
rows=[]
scenes=[(name,ROOT/'lrtest/in'/(name+'.tif'),[c[k] for k in ('rx','ry','rz')],parse_adobe(ROOT/'adobe_truth'/f"img_{c['text']}.lua")[1]) for name,c in CASES.items()]
scenes += [(name,ROOT/'lrtest'/folder/(name+'.tif'),[M.PITCH15,M.YAW6,0.],None) for name,folder in [('NM_single_wall','nm'),('NM_railing','nm'),('NM_curves','nm'),('NM_three_plane','nm'),('CT_manhattan2','nm2')]]
for name,path,pose,Ha in scenes:
 image=cv2.imread(str(path),0);h,w=image.shape;segments=up.detect_segments(image)
 vs=visible_verticals(pose,1600,w,h);conv=convergence(np.eye(3),vs,w,h);gt=M.ground_truth_H(*pose);base=sim_residual(np.eye(3),gt,grid)[0]
 for seed in range(5):
  for label in labels:
   up._auto_pitch_scale=candidate(label)
   out=up.solve_ex(segments,w,h,1600,'auto',seed=seed)
   row=dict(scene=name,seed=seed,variant=label,ok=out['ok'],reason=out['reason'],damping=out['damping'],ideal_fraction=ideal_fraction(out['H'],pose),removed=1-convergence(out['H'],vs,w,h)/conv)
   if Ha is not None:row.update(agreement(Ha,out['H'],w,h))
   rows.append(row)
 print('completed',name,flush=True)
up._auto_pitch_scale=original
summary={}
for name,_,_,_ in scenes:
 summary[name]={}
 for label in labels:
  rs=[r for r in rows if r['scene']==name and r['variant']==label]
  fields=['ideal_fraction','removed']+(['raw_rms_pct','shape_rms_pct'] if 'raw_rms_pct' in rs[0] else [])
  summary[name][label]={k:float(np.mean([r[k] for r in rs])) for k in fields}
(ROOT/'evidence/round5_ablations.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n')
print(json.dumps(summary,indent=2))
