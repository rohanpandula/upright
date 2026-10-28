"""Paired summaries of Round 11 input interventions, including all failures."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evidence'))
from round10_analyze import distribution

def load(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
BASE={p['image_id']:p for p in load(ROOT/'evidence/round10_candidates.jsonl')}
ACTUAL={p['image_id']:p for p in load(ROOT/'evidence/round10_actual.jsonl')}

def baseline(ids):
    out=[]
    for iid in ids:
        p=BASE[iid];a=ACTUAL[iid];rows=[]
        for r,t in zip(p['rows'],a['rows']):
            rows.append(dict(mode=r['mode'],fit=r['candidate_ok'],accepted=t['candidate_ok'],identity=r['identity'],candidate=r['candidate'],returned=t['candidate'] if t['candidate_ok'] else r['identity'],rotation=r['rotation'],damping=r['damping'],source_focal_px=r['source_focal_px'],quality=r['quality']))
        out.append(dict(image_id=iid,rows=rows))
    return out

def summarize(photos):
    result=dict(photos=len(photos),errors=sum('error' in p for p in photos))
    for mode in ['auto','full']:
        rows=[r for p in photos for r in p.get('rows',[]) if r['mode']==mode]
        stats=dict(n=len(rows),fit=sum(r['fit'] for r in rows),accepted=sum(r['accepted'] for r in rows))
        for variant in ['candidate','returned']:
            values=[];delta=[];win=lose=tie=invalid=0
            for r in rows:
                e=(r[variant] or {}).get('raw_rms_pct')
                if variant=='candidate' and not r['fit']:e=r['identity']['raw_rms_pct']
                if e is None:invalid+=1;continue
                identity=r['identity']['raw_rms_pct'];values.append(e);delta.append(e-identity)
                if e<identity-1e-9:win+=1
                elif e>identity+1e-9:lose+=1
                else:tie+=1
            stats[variant]=dict(win=win,lose=lose,tie=tie,invalid=invalid,win_all_pct=100*win/len(rows) if rows else None,win_decisive_pct=100*win/(win+lose) if win+lose else None,error=distribution(values) if values else None,delta_median=float(np.median(delta)) if delta else None)
        result[mode]=stats
    return result

def paired(a,b):
    aa={p['image_id']:p for p in a};bb={p['image_id']:p for p in b};result={}
    for mode in ['auto','full']:
        result[mode]={}
        for v in ['candidate','returned']:
            win=lose=tie=invalid=0;diff=[]
            for iid in aa.keys()&bb.keys():
                if 'error' in aa[iid] or 'error' in bb[iid]:continue
                x=next(r for r in aa[iid]['rows'] if r['mode']==mode);y=next(r for r in bb[iid]['rows'] if r['mode']==mode)
                def value(r):
                    if v=='candidate' and not r['fit']:return r['identity']['raw_rms_pct']
                    return (r[v] or {}).get('raw_rms_pct')
                xx,yy=value(x),value(y)
                if xx is None or yy is None:invalid+=1;continue
                d=yy-xx;diff.append(d)
                if d< -1e-9:win+=1
                elif d>1e-9:lose+=1
                else:tie+=1
            result[mode][v]=dict(improve=win,worsen=lose,tie=tie,invalid=invalid,median_delta=float(np.median(diff)) if diff else None)
    return result

def analyze(path):
    data=load(path);variants=sorted({p['variant'] for p in data});result={}
    for variant in variants:
        photos=[p for p in data if p['variant']==variant];ids=[p['image_id'] for p in photos if 'error' not in p];b=baseline(ids)
        good=[p for p in photos if 'error' not in p]
        result[variant]=dict(variant=summarize(photos),paired_raw=summarize(b),vs_raw=paired(b,good))
        stats=[p['info']['stats'] for p in good if 'stats' in p.get('info',{})]
        result[variant]['image_stats_median']={k:np.median([s[k] for s in stats],axis=0).tolist() for k in stats[0]} if stats else {}
        complete=[]
        for p in photos:
            if 'error' not in p:
                complete.append(p);continue
            b=baseline([p['image_id']])[0]
            if variant=='preview':
                complete.append(dict(b,error=p['error']))
            else:
                rows=[dict(r,fit=False,accepted=False,candidate=None,returned=r['identity']) for r in b['rows']]
                complete.append(dict(image_id=p['image_id'],rows=rows,error=p['error']))
        result[variant]['failure_policy']='raw fallback' if variant=='preview' else 'identity on experimental failure'
        result[variant]['complete']=summarize(complete)
        result[variant]['complete_raw']=summarize(baseline([p['image_id'] for p in photos]))
    return result

if __name__=='__main__':
    r=analyze(sys.argv[1]);out=Path(sys.argv[1]).with_suffix('.summary.json');out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
