"""Score selected supports after fitting. Native truth never enters either solve."""
from pathlib import Path
import concurrent.futures
import importlib.util
import json
import sys
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'port'),str(ROOT/'evidence')]
from round9_probe import json_default


def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

BEFORE=module('before','evidence/round12_before_port.py')
AFTER=module('candidate','evidence/round12_candidate_port.py')


def measure(r):
    cv2.setNumThreads(1)
    z=np.load(ROOT/f'evidence/round10_supports/{r["image_id"]}.npz');seg,Q=z['seg'],z['Q']
    sw,sh=int(r['width']),int(r['height'])
    w,h=(sh,sw) if r['orientation'] in ('BC','DA','AD','CB') else (sw,sh)
    results={}
    for name,up in [('before',BEFORE),('candidate',AFTER)]:
        captured=[];quality=up._alignment_quality
        def capture(H,v,g,w,h):
            captured.append([v.copy(),g.copy()]);return dict(quality(H,v,g,w,h),passed=True)
        focal=up.metadata_focal_px(r['meta'].get('UprightFocalLength35mm'),w,h)
        with patch.object(up,'_alignment_quality',side_effect=capture):
            out=up.solve_ex(seg,w,h,focal,'full',framing=True,scaling=True)
        record=dict(map=out['ok'],reason=out['reason'],rotation=out['rotation'],families=[])
        # Only now read and apply the diagnostic reference transform.
        if captured:
            native=Q@np.array(r['transform']['2']).reshape(3,3)@np.linalg.inv(Q)
            for k,s in enumerate(captured[-1]):
                p=np.c_[s.reshape(-1,2),np.ones(2*len(s))]@native.T
                p=(p[:,:2]/p[:,2:]*[w,h]).reshape(-1,4)
                delta=p[:,2:]-p[:,:2]
                a=np.degrees(np.arctan2(abs(delta[:,1]),abs(delta[:,0])))
                good=a>=89 if k==0 else a<=1
                length=np.linalg.norm((s[:,2:]-s[:,:2])*[w,h],axis=1)
                record['families'].append(dict(count=len(s),good_count=int(good.sum()),
                    length_weighted_purity=float(length[good].sum()/length.sum())))
        results[name]=record
    return dict(image_id=r['image_id'],split=r['split'],**results)

if __name__=='__main__':
    rows=json.loads((ROOT/'evidence/round11_sample.json').read_text())
    rows=[r for r in rows if Path(r['path']).suffix.lower() in {'.arw','.dng','.nef','.raf'}]
    with (ROOT/'evidence/round12_support_audit.jsonl').open('w') as f,concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for i,r in enumerate(pool.map(measure,rows),1):
            f.write(json.dumps(r,default=json_default)+'\n');f.flush()
            if i%50==0 or i==len(rows):print('supports',i,'/',len(rows),flush=True)
