"""Calibration tables and untouched validation; all distances use sensor dimensions."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def load(name):
    return [json.loads(x) for x in (ROOT/'evidence'/name).read_text().splitlines()]


def attempts(photos):
    return [dict(r,image_id=p['image_id'],path=p['path'],cohort=p['cohort'],split=p['split'])
            for p in photos for r in p.get('rows',[])]


def decision(r,scatter,bias):
    q=r['quality']
    if not q:return False
    if bias=='old':
        bias_pass=max(q['family_bias_bound_deg'])<=.1
    elif bias=='excess':
        bias_pass=max(np.abs(q['family_signed_median_deg'])-np.array(q['family_median_margin_deg']))<=.1
    elif bias=='resolved':
        bias_pass=bool(np.all(np.abs(q['family_signed_median_deg'])<=np.maximum(.1,q['family_median_margin_deg'])))
    elif bias.startswith('relative_'):
        fraction=float(bias.split('_')[1])
        bias_pass=bool(np.all((np.array(q['family_bias_bound_deg'])<=.1)|(np.array(q['family_bias_bound_deg'])<fraction*np.array(q['family_median_deg']))))
    elif bias=='none':
        bias_pass=True
    elif bias=='count':
        # Direct 1/sqrt(N) scaling of the absolute median: endpoint-noise-only comparator.
        bias_pass=True
        return bool(max(np.array(q['family_median_deg'])*np.sqrt(q['family_effective_n']))<=scatter)
    else:raise ValueError(bias)
    return max(q['family_median_deg'])<=scatter and bool(bias_pass)


def distribution(values):
    a=np.array(values,float)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p90=float(np.percentile(a,90)),p99=float(np.percentile(a,99)),max=float(a.max()),
                **{f'over{t}':int((a>t).sum()) for t in (1,5,20)})


def matrix(rows,passes,threshold):
    counts=dict(accept_good=0,accept_bad=0,reject_good=0,reject_bad=0,no_candidate=0)
    for r,p in zip(rows,passes):
        if r['quality'] is None:
            counts['no_candidate']+=1;continue
        error=r['candidate']['raw_rms_pct'] if r['candidate'] is not None else None
        bad=error is None or error>threshold
        counts[('accept_' if p else 'reject_')+('bad' if bad else 'good')]+=1
    return counts


def summary(rows,scatter,bias):
    passes=[decision(r,scatter,bias) for r in rows]
    returned=[]
    final_passes=[]
    for r,p in zip(rows,passes):
        use=p and r['candidate_ok'] and r['candidate'] is not None and r['candidate']['raw_rms_pct'] is not None
        final_passes.append(use)
        returned.append(r['candidate']['raw_rms_pct'] if use else r['identity']['raw_rms_pct'])
    native={k:0 for k in ('accept_corrected','accept_abstained','reject_corrected','reject_abstained')}
    for r,p in zip(rows,passes):
        native[('accept_' if p else 'reject_')+('abstained' if r['adobe_abstained'] else 'corrected')]+=1
    return dict(gate_accept=sum(passes),returned_accept=sum(final_passes),
                fit_count=sum(r['quality'] is not None for r in rows),
                error=distribution(returned),
                confusion={str(t):matrix(rows,passes,t) for t in (1,5,20)},vs_adobe=native)


def calibration():
    rows=[r for r in attempts(load('round10_candidates.jsonl')) if r['split']=='calibration']
    result=[]
    for bias in ('old','excess','resolved','relative_0.25','relative_0.5','relative_0.75','relative_1.0','none','count'):
        thresholds=(.2,.5,.75,1.,1.5,2.,3.,5.,10.,90.) if bias!='count' else (1.,3.,5.,10.,20.)
        for scatter in thresholds:
            result.append(dict(scatter=scatter,bias=bias,**summary(rows,scatter,bias)))
    (ROOT/'evidence/round10_calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    print('calibration attempts',len(rows))
    print('bias scatter gate_accept pass_good<=1 pass_bad>5 returned_median p90 >5 max')
    for r in result:
        e=r['error'];print(r['bias'],r['scatter'],r['gate_accept'],r['confusion']['1']['accept_good'],
                r['confusion']['5']['accept_bad'],*[round(e[k],5) for k in ('median','p90','over5','max')])


if __name__=='__main__':calibration()
