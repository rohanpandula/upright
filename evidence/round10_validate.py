"""Fresh synthetic nonregression checks, with immutable round-9 implementation."""
from pathlib import Path
import argparse
import ast
import importlib.util
import inspect
import json
import sys
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'port'),str(ROOT/'evidence'),str(ROOT)]
import upright_port as up
import benchmark_accuracy as bench
from benchmark_adobe import agreement
from round9_probe import holdout, sweep as quad_sweep, boundary_controls
from round8_probe import measure as adobe_scaled
from round10_analyze import distribution

spec=importlib.util.spec_from_file_location('round10_before',ROOT/'evidence/round10_before_port.py')
before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)


def sweep():
    rows=[]
    for scene,pose in bench.CASES.items():
        image=cv2.imread(str(ROOT/'lrtest/in'/f'{scene}.tif'),0);h,w=image.shape
        seg=up.detect_segments(image)
        np.testing.assert_array_equal(seg,before.detect_segments(image))
        truth=bench.ground_truth(pose,1600,w,h,9,-5,20)
        for seed in range(40):
            for mode in ('auto','full','level','vertical'):
                old=before.solve_ex(seg,w,h,1600,mode,seed=seed,framing=True,scaling=True)
                out=up.solve_ex(seg,w,h,1600,mode,seed=seed,framing=True,scaling=True)
                assert old['ok'] and out['ok'],(scene,seed,mode,out['reason'])
                for key in ('H','H_unscaled','H_unframed'):
                    np.testing.assert_array_equal(old[key],out[key])
                rows.append(dict(scene=scene,seed=seed,mode=mode,ok=out['ok'],
                    matrix_maxdiff=float(np.max(abs(old['H']-out['H']))),
                    axis=bench.deviation(out['H'],truth,np.array([0,0,1,1]),w,h),quality=out['quality']))
        print('40-seed all-mode nonregression',scene,flush=True)
    return rows


def bad_replay():
    image=cv2.imread(str(ROOT/'lrtest/in/D_roll.tif'),0);h,w=image.shape
    seg=up.detect_segments(image)
    # Reconstruct the actual pre-round1 lateral selection, not repaired families.
    source=inspect.getsource(up._base_solve)
    score='return abs(_normalize(K0i @ v["vp"])[0])'
    assert score in source
    source=source.replace(score,'p = v["vp"]\n        n = float(np.hypot(p[0], p[1]))\n        return abs(p[0]) / n if n > _EPS else 0.0')
    namespace=dict(up.__dict__);exec(source,namespace)
    # Round1 recorded theta and original supports isolate the gate policy.
    theta=np.array(json.loads((ROOT/'evidence/round1_diagnosis.json').read_text())['seed4']['theta'])
    optimizer=up._levenberg_marquardt
    def replay(fun,x0,lo,hi,**kwargs):
        if len(x0)==4:return optimizer(fun,x0,lo,hi,**kwargs)
        return theta,0.
    namespace['_levenberg_marquardt']=replay
    namespace['_source_calibration']=lambda v,g,vp,gh,w,h,prior:(prior,None)
    rows=[]
    with patch.object(up,'_base_solve',namespace['_base_solve']):
        for mode in ('auto','full','level','vertical'):
            out=up.solve_ex(seg,w,h,1600,mode,seed=4,framing=True,scaling=True)
            assert not out['ok'] and not out['quality']['passed']
            np.testing.assert_array_equal(out['H'],np.eye(3))
            rows.append(dict(mode=mode,quality=out['quality'],reason=out['reason']))
    return rows


def holdout_check():
    rows=holdout()
    old=json.loads((ROOT/'evidence/round9_heldout.json').read_text())
    assert len(rows)==len(old)==280
    for a,b in zip(rows,old):
        assert a['ok'] and b['ok']
        assert (a['scene'],a['mode'],a['seed'])==(b['scene'],b['mode'],b['seed'])
        np.testing.assert_array_equal(a['H'],b['H'])
    return rows


def adobe_check():
    rows=adobe_scaled()
    old=json.loads((ROOT/'evidence/round8.json').read_text())['measurements']
    assert len(rows)==len(old)==160
    for a,b in zip(rows,old):
        np.testing.assert_array_equal(a['H'],b['H'])
        assert a['final']==b['final']
    print('Adobe framing+scale scores unchanged',flush=True)
    for prior in ('scene_focal','adobe_stored_focal'):
        subset=[r['final']['raw_rms_pct'] for r in rows if r['prior']==prior]
        print(prior,distribution(subset),flush=True)
    return rows


def scope():
    old=ast.parse((ROOT/'evidence/round10_before_port.py').read_text())
    new=ast.parse((ROOT/'port/upright_port.py').read_text())
    get=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
    a,b=get(old),get(new)
    changed=[name for name in a if a[name]!=b[name]]
    assert changed==['_alignment_quality']
    globals=lambda t:[ast.dump(n,include_attributes=False) for n in t.body if not isinstance(n,ast.FunctionDef)]
    assert globals(old)==globals(new)
    return dict(changed_functions=changed,unchanged_functions=len(a)-1)


def geometry_check():
    result=quad_sweep()
    old=json.loads((ROOT/'evidence/round9_sweep.json').read_text())
    assert result==old
    result['boundary_controls']=boundary_controls()
    print('57,036 band cases and all stress/boundary controls unchanged',flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('sweep','bad','heldout','adobe','scope','geometry'))
    args=ap.parse_args()
    result={'sweep':sweep,'bad':bad_replay,'heldout':holdout_check,'adobe':adobe_check,'scope':scope,'geometry':geometry_check}[args.stage]()
    (ROOT/f'evidence/round10_{args.stage}.json').write_text(json.dumps(result,indent=2)+'\n')
