"""MIT. Reproducible fixed-source interventions and Round 2 comparison."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'port'))
import json, hashlib
from unittest.mock import patch
import cv2
import numpy as np
import upright_port as up
import benchmark_accuracy as bench


def retain_prior(*args):
    return args[-1],dict(accepted=False,reason='experiment: fixed source')


def diagnosis():
    im=cv2.imread(str(bench.ROOT/'01_input.png'),0);h,w=im.shape
    seg=up.detect_segments(im)
    truth=bench.ground_truth((-.38,.10,.04),1400,w,h,8,-5,18)
    kind=np.array([0,0,1,1])
    vps=up.estimate_vanishing_points(seg,w,h)
    v,hs=up._classify(vps,seg/[w,h,w,h])
    Kp=np.array([[1200,0,w/2],[0,1200,h/2],[0,0,1]])
    Ki=np.linalg.inv(Kp)
    hs.sort(key=lambda a:abs(up._normalize(Ki@a['vp'])[0]),reverse=True)
    a,b=v['vp'].copy(),hs[0]['vp'].copy()
    ac=a[:2]-np.array([w/2,h/2])*a[2]
    bc=b[:2]-np.array([w/2,h/2])*b[2]
    direct_f2=-float(ac@bc)/(a[2]*b[2])
    with np.load(bench.SEGMENTS) as z:
        legacy=np.r_[z['legacy_v'],z['legacy_h']]
        legacy_kind=np.r_[np.zeros(len(z['legacy_v'])),np.ones(len(z['legacy_h']))]
    lm=up._levenberg_marquardt; energy=up._energy_residuals
    rows={}
    for label in ('fixed_source','no_horizon','fixed_destination_centre','recalibrated'):
        fits=[]
        def optimize(fun,x0,lo,hi,**kw):
            if len(x0)==6 and label in ('fixed_source','no_horizon'):
                lo=lo.copy();hi=hi.copy();lo[4:]=.2;hi[4:]=.8
            theta,cost=lm(fun,x0,lo,hi,**kw)
            fits.append(dict(start=x0.tolist(),theta=theta.tolist(),cost=cost))
            return theta,cost
        def objective(*args):
            if label=='no_horizon':
                args=list(args);args[7]=dict(args[7],w_eye=0)
            return energy(*args)
        with patch.object(up,'_levenberg_marquardt',side_effect=optimize), \
             patch.object(up,'_energy_residuals',side_effect=objective), \
             patch.object(up,'_source_calibration',side_effect=(
                 up._source_calibration if label=='recalibrated' else retain_prior)):
            o=up.solve_ex(seg,w,h,1200,'full')
        rows[label]=dict(ok=o['ok'],reason=o['reason'],fits=fits,
            quality=o['quality'],source_focal_px=o['source_focal_px'],
            truth_axis_deg=bench.deviation(o['H'],truth,kind,w,h),
            legacy_axis_deg=bench.deviation(o['H'],legacy,legacy_kind,w,h))
    rows['focal_sweep']=[]
    for focal in (980,1200,1400,1820):
        for seed in range(5):
            o=up.solve_ex(seg,w,h,focal,'full',seed=seed)
            rows['focal_sweep'].append(dict(focal=focal,seed=seed,ok=o['ok'],
                source_focal_px=o['source_focal_px'],calibration=o['calibration'],
                truth_axis_deg=bench.deviation(o['H'],truth,kind,w,h),
                legacy_axis_deg=bench.deviation(o['H'],legacy,legacy_kind,w,h)))
    rows['initial_geometry']=dict(tls_pair_focal_px=float(np.sqrt(direct_f2)),
        prior_ray_dot=float(up._normalize(Ki@a)@up._normalize(Ki@b)))
    return rows


def fixtures():
    rows={}
    for name,pose in [*bench.CASES.items(),('01_input_truth',(-.38,.10,.04))]:
        demo=name=='01_input_truth'; focal=1400 if demo else 1600
        im=cv2.imread(str(bench.ROOT/('01_input.png' if demo else 'lrtest/in/'+name+'.tif')),0)
        h,w=im.shape;s=up.detect_segments(im)
        t=bench.ground_truth(pose,focal,w,h,8 if demo else 9,-5,18 if demo else 20)
        rows[name]=[]
        for seed in range(5):
            o=up.solve_ex(s,w,h,focal,'full',seed=seed)
            rows[name].append(dict(seed=seed,ok=o['ok'],calibration=o['calibration'],
                median_axis_deg=bench.deviation(o['H'],t,np.array([0,0,1,1]),w,h)))
    return rows


def focal_stress():
    sys.path.insert(0,str(bench.ROOT))
    import make_heldout_images as heldout
    rows={}
    for case in heldout.CASES:
        image,truth,kind,_=heldout.render(case)
        w,h=case['w'],case['h'];f=heldout.camera(case)[1][0,0]
        segments=up.detect_segments(image);rows[case['name']]=[]
        for factor in (.7,1.3):
            with patch.object(up,'_source_calibration',side_effect=retain_prior):
                old=up.solve_ex(segments,w,h,f*factor,'full')
            new=up.solve_ex(segments,w,h,f*factor,'full')
            rows[case['name']].append(dict(metadata_factor=factor,
                old_ok=old['ok'],new_ok=new['ok'],calibration=new['calibration'],
                old_axis_deg=bench.deviation(old['H'],truth,kind,w,h),
                new_axis_deg=bench.deviation(new['H'],truth,kind,w,h)))
    return rows


if __name__=='__main__':
    out=dict(diagnosis=diagnosis(),fixture_seeds=fixtures(),accuracy=bench.run(),
             heldout_focal_stress=focal_stress())
    held=json.loads((bench.ROOT/'evidence/round2_heldout.json').read_text())
    for n,names in [('four',list(bench.CASES)),('five',list(out['fixture_seeds']))]:
        scores=[r['median_axis_deg'] for name in names for r in out['fixture_seeds'][name]]
        mean=float(np.mean(scores));worst=float(np.max(scores))
        hs=held['mean_family_deg']
        out[n+'_comparison']=dict(fixture_mean_deg=mean,fixture_worst_deg=worst,
            heldout_mean_deg=hs,heldout_minus_fixture_deg=hs-mean,
            mean_change_percent=100*(hs/mean-1),heldout_worst_deg=held['worst_family_deg'])
    out['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (bench.ROOT/'make_heldout_images.py',bench.ROOT/'make_test_images.py',
         bench.ROOT/'port/upright_port.py')}
    (bench.ROOT/'evidence/round2_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k.endswith('comparison')},indent=2))
