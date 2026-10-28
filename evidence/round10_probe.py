"""Round 10 real-photo calibration. Oracle transforms are used only for scoring."""
from pathlib import Path
import argparse
import concurrent.futures
import importlib.util
import json
import random
import sqlite3
import sys
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'port'), str(ROOT/'evidence'), str(ROOT)]
import upright_port as up
from benchmark_adobe import agreement
from round9_probe import load_photo, json_default
from harvest_lr_transforms import QUERY, parse_text
CACHE = ROOT/'evidence/round10_supports'
_spec = importlib.util.spec_from_file_location('round10_before', ROOT/'evidence/round10_before_port.py')
_before = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_before)


def sample():
    main = json.loads((ROOT/'evidence/round9_sample.json').read_text())
    abstain = json.loads((ROOT/'evidence/real_photos_abstain_sample.json').read_text())
    for i,r in enumerate(main):
        r.update(cohort='round9', split='calibration' if i < 150 else 'validation')
    for i,r in enumerate(abstain):
        r.update(cohort='abstain', split='calibration' if i < 114 else 'validation')
    seen = {r['path'] for r in main+abstain}
    available, repro = [], []
    with sqlite3.connect('file:/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat?mode=ro',uri=True) as con:
        con.execute('PRAGMA query_only=ON')
        for iid,text,base,ext,folder,root,w,h,ori in con.execute(QUERY):
            p = parse_text(text); tr = p.pop('transforms')
            path = str(Path(root)/folder/f'{base}.{ext}')
            if not Path(path).is_file() or '/research/upright/' in path:
                continue
            if Path(path).suffix.lower() in ('.mp4','.mov'):
                continue
            r = dict(image_id=iid,path=path,width=w,height=h,orientation=ori,
                     transform={str(k):v for k,v in tr.items()},meta=p)
            if base in ('DSCF3578','DSCF3582','DSCF3586','DSCF3590'):
                repro.append(dict(r,cohort='repro',split='repro'))
            if path in seen or any(k not in tr or np.max(abs(np.array(tr[k]).reshape(3,3)-np.eye(3)))<1e-6 for k in (1,2)):
                continue
            available.append(r)
    unique = {r['path']:r for r in available}
    extra = random.Random(10010).sample(sorted(unique.values(),key=lambda r:r['image_id']),150)
    for r in extra:
        r.update(cohort='new_holdout',split='validation')
    rows = main+abstain+extra
    assert len({r['path'] for r in rows}) == len(rows) == 679
    (ROOT/'evidence/round10_sample.json').write_text(json.dumps(rows,indent=2)+'\n')
    (ROOT/'evidence/round10_repro_sample.json').write_text(json.dumps(repro,indent=2)+'\n')
    print('frozen',len(rows),'photos; repro',len(repro),flush=True)


def supports(r):
    CACHE.mkdir(exist_ok=True)
    path = CACHE/f"{r['image_id']}.npz"
    if path.exists():
        z = np.load(path)
        return z['seg'],z['Q'],json.loads(str(z['decode']))
    gray,Q,decode = load_photo(r)
    wh,ww = gray.shape
    w,h = int(r['width']),int(r['height'])
    if r['orientation'] in ('BC','DA','AD','CB'):
        w,h = h,w
    seg = up.detect_segments(gray)*[w/ww,h/wh,w/ww,h/wh]
    np.savez_compressed(path,seg=seg,Q=Q,decode=json.dumps(decode))
    return seg,Q,decode


def measure(r, bypass=True):
    cv2.setNumThreads(1)
    try:
        seg,Q,decode = supports(r)
    except Exception as exc:
        return dict(image_id=r['image_id'],cohort=r['cohort'],split=r['split'],path=r['path'],error=repr(exc))
    sw,sh = int(r['width']),int(r['height'])
    w,h = (sh,sw) if r['orientation'] in ('BC','DA','AD','CB') else (sw,sh)
    f = up.metadata_focal_px(r['meta'].get('UprightFocalLength35mm'),w,h)
    # Preserve pre-change diagnostics when reproducing ungated calibration.
    original = _before._alignment_quality if bypass else up._alignment_quality
    def capture(H,v,g,w,h):
        q = original(H,v,g,w,h)
        q['family_support_count'] = [len(v),len(g)]
        lengths = [np.hypot((a[:,2]-a[:,0])*w,(a[:,3]-a[:,1])*h) for a in (v,g)]
        q['family_effective_n'] = [float(a.sum()**2/(a@a)) for a in lengths]
        if bypass:
            q = dict(q,passed=True)
        return q
    rows = []
    for mode,slot in (('auto','1'),('full','2')):
        native = np.array(r['transform'][slot]).reshape(3,3)
        with patch.object(up,'_alignment_quality',side_effect=capture):
            candidate = up.solve_ex(seg,w,h,f,mode,seed=0,framing=True,scaling=True)
        def score(H):
            try:
                return agreement(native,np.linalg.inv(Q)@H@Q,sw,sh)
            except ValueError as exc:
                return dict(raw_rms_pct=None,error=str(exc))
        q = candidate['quality']
        rows.append(dict(mode=mode,quality=q,candidate_ok=candidate['ok'],reason=candidate['reason'],
            candidate=score(candidate['H']) if candidate['ok'] else None,
            bare=score(candidate.get('H_unframed',candidate['H'])) if candidate['ok'] else None,
            identity=score(np.eye(3)),H=candidate['H'].tolist(),
            adobe_abstained=bool(np.max(abs(native-np.eye(3)))<1e-6),
            rotation=candidate['rotation'],damping=candidate['damping'],
            source_focal_px=candidate['source_focal_px'],calibration=candidate['calibration']))
    return dict(image_id=r['image_id'],path=r['path'],cohort=r['cohort'],split=r['split'],
                width=w,height=h,sensor_width=sw,sensor_height=sh,decode=decode,segments=len(seg),rows=rows)


def collect(workers, actual=False, repro=False):
    sample_path = ROOT/('evidence/round10_repro_sample.json' if repro else 'evidence/round10_sample.json')
    rows = json.loads(sample_path.read_text())
    name = 'repro' if repro else ('actual' if actual else 'candidates')
    with (ROOT/f'evidence/round10_{name}.jsonl').open('w') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i,result in enumerate(pool.map(measure,rows,[not actual]*len(rows)),1):
                f.write(json.dumps(result,default=json_default)+'\n');f.flush()
                if i%20==0 or i==len(rows):
                    print(name,i,'/',len(rows),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('sample','collect','actual','repro'))
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    if args.stage=='sample':sample()
    else:collect(args.workers,actual=args.stage=='actual',repro=args.stage=='repro')
