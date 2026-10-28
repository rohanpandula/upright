"""Per-seed gate audit on independent world lines. Python 3.12; MIT.

Candidate scores use the undamped rotation, including rejected fits. Returned
identity scores are retained separately; rejection cannot improve accuracy by
silently dropping an attempt. No truth is supplied to fitting or gating.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import benchmark_accuracy as bench
import upright_port as up
sys.path.insert(0, str(bench.ROOT))
import make_heldout_images as heldout


def confusion(trials):
    counts = dict(pass_good=0, pass_bad=0, reject_good=0, reject_bad=0, no_fit=0)
    for t in trials:
        if t['candidate_axis_deg'] is None:
            counts['no_fit'] += 1
        else:
            key = ('pass_' if t['gate_pass'] else 'reject_')
            key += 'bad' if max(t['candidate_axis_deg']) > .1 else 'good'
            counts[key] += 1
    return counts


def run(seeds=range(50)):
    scenes = {}
    fixtures = [(name, bench.ROOT/'lrtest/in'/(name+'.tif'), pose, 1600, 9, -5, 20)
                for name, pose in bench.CASES.items()]
    fixtures += [('01_input_'+str(focal), bench.ROOT/'01_input.png',
                  (-.38,.10,.04), focal, 8,-5,18) for focal in (1400,980,1200,1820)]
    inputs = []
    for name,path,pose,focal,x,ylo,yhi in fixtures:
        image=cv2.imread(str(path),0)
        h,w=image.shape
        truth=bench.ground_truth(pose,1400 if name.startswith('01_') else focal,w,h,x,ylo,yhi)
        inputs.append((name,'fixtures',image,truth,np.array([0,0,1,1]),focal))
    for c in heldout.CASES:
        image,truth,kind,_=heldout.render(c)
        inputs.append((c['name'],'heldout',image,truth,kind,float(heldout.camera(c)[1][0,0])))
    for name,group,image,truth,kind,focal in inputs:
        h,w=image.shape
        segments=up.detect_segments(image)
        trials=[]
        for seed in seeds:
            out=up.solve_ex(segments,w,h,focal,'full',seed=seed)
            candidate=bench.candidate_deviation(out,truth,kind,w,h)
            trials.append(dict(seed=seed,ok=out['ok'],reason=out['reason'],
                gate_pass=bool(out['quality'] and out['quality']['passed']),
                candidate_axis_deg=candidate,
                returned_axis_deg=bench.deviation(out['H'],truth,kind,w,h),
                quality=out['quality'],source_focal_px=out['source_focal_px'],
                calibration=out['calibration']))
        scenes[name]=dict(group=group,trials=trials)
        print(name,confusion(trials),flush=True)
    return dict(protocol='Full; detector once per raster; seeds '+str(list(seeds))+
                '; bad = max(independent V/H median axis deg) > 0.1; '
                'confusion uses undamped candidates, returned identities scored separately',
                scenes=scenes,confusion={group:confusion([t for s in scenes.values()
                    if s['group']==group for t in s['trials']]) for group in ('fixtures','heldout')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,default=50)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.seeds<1:parser.error('--seeds must be positive')
    result=run(range(args.seeds))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['confusion'])
