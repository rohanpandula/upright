"""Frozen support replay; Adobe maps enter scoring only, never selection."""
from pathlib import Path
import argparse
import concurrent.futures
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
import round11_probe as probe
from round9_probe import json_default


def measure(r):
    cv2.setNumThreads(1)
    try:
        z=np.load(ROOT/f'evidence/round10_supports/{r["image_id"]}.npz')
        return dict(image_id=r['image_id'],cohort=r['cohort'],split=r['split'],
                    rows=probe.solve_rows(r,z['seg'],z['Q']))
    except Exception as exc:
        return dict(image_id=r['image_id'],split=r['split'],error=repr(exc))


def run():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--calibration',action='store_true');ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args()
    sample=json.loads((ROOT/'evidence/round10_sample.json').read_text())
    if a.calibration:sample=[r for r in sample if r['split']=='calibration' and r['cohort']!='abstain']
    if a.limit:sample=sample[:a.limit]
    with (ROOT/f'evidence/round12_{a.label}.jsonl').open('w') as fh,concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(measure,sample),1):
            fh.write(json.dumps(r,default=json_default)+'\n');fh.flush()
            if i%25==0 or i==len(sample):print(a.label,i,'/',len(sample),flush=True)

if __name__=='__main__':run()
