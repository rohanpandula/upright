"""Failure-retaining summaries for selector experiments; no fitting here."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evidence'))
import round11_analyze as a


def main():
    metadata={r['image_id']:r for r in json.loads((ROOT/'evidence/round10_sample.json').read_text())}
    final=a.load(ROOT/'evidence/round12_final_candidate_all.jsonl')
    first=a.load(ROOT/'evidence/round12_candidate_all.jsonl')
    assert len(final)==len(first)==679
    errors=[r for r in final if 'error' in r]
    assert len(errors)==1 and errors[0]['image_id']==2116098
    groups={}
    for name,predicate in {
        'raw387':lambda m:m['cohort']!='abstain' and Path(m['path']).suffix.lower() in {'.arw','.dng','.nef','.raf'},
        'corrected450':lambda m:m['cohort']!='abstain',
        'raw_calibration':lambda m:m['cohort']!='abstain' and m['split']=='calibration' and Path(m['path']).suffix.lower() in {'.arw','.dng','.nef','.raf'},
        'raw_validation':lambda m:m['cohort']!='abstain' and m['split']=='validation' and Path(m['path']).suffix.lower() in {'.arw','.dng','.nef','.raf'},
        'abstention_decoded':lambda m:m['cohort']=='abstain',
    }.items():
        p=[r for r in final if 'error' not in r and predicate(metadata[r['image_id']])]
        ids={r['image_id'] for r in p};base=a.baseline(ids)
        entry=dict(before=a.summarize(base),candidate=a.summarize(p),paired=a.paired(base,p),
                   initial_candidate=a.summarize([r for r in first if r['image_id'] in ids]))
        if name=='raw387':
            assert len(p)==387
            for mode in ['auto','full']:
                entry[mode+'_common_maps']={}
                for version,data in [('before',base),('candidate',p)]:
                    lookup={r['image_id']:next(t for t in r['rows'] if t['mode']==mode) for r in data}
                    other={r['image_id']:next(t for t in r['rows'] if t['mode']==mode) for r in (p if version=='before' else base)}
                    common=[t['candidate']['raw_rms_pct'] for iid,t in lookup.items() if t['fit'] and other[iid]['fit']]
                    entry[mode+'_common_maps'][version]=dict(n=len(common),median=float(np.median(common)))
        groups[name]=entry
    oracle=json.loads((ROOT/'evidence/round11_oracle_all.summary.json').read_text())['oracle_labels']['complete']
    before=groups['raw387']['before']['full']['candidate']['error']['median']
    after=groups['raw387']['candidate']['full']['candidate']['error']['median']
    ceiling=oracle['full']['candidate']['error']['median']
    audit=a.load(ROOT/'evidence/round12_support_audit.jsonl');assert len(audit)==387
    diagnostic={}
    for name in ['before','candidate']:
        selected=[r[name] for r in audit if r[name]['rotation'] is not None]
        purity=np.array([[f['length_weighted_purity'] for f in r['families']] for r in selected])
        assert np.isfinite(purity).all()
        diagnostic[name]=dict(selected=len(selected),maps=sum(r[name]['map'] for r in audit),
            horizon=sum('horizon' in r[name]['reason'] for r in audit),
            rotation_norm_p50_p90_deg=np.percentile([np.linalg.norm(r['rotation'])*180/np.pi for r in selected],[50,90]).tolist(),
            purity_median_VH=np.median(purity,axis=0).tolist(),both_purity_ge80=int(np.sum(np.min(purity,axis=1)>=.8)))
    variants={}
    for label in ['no_axis']+[f'geometry_v{i}' for i in range(1,10)]:
        p=a.load(ROOT/f'evidence/round12_{label}.jsonl');assert len(p)==150
        variants[label]=a.summarize(p)
    result=dict(groups=groups,errors=errors,oracle=oracle,diagnostic=diagnostic,calibration_variants=variants,
        full_gap=dict(before=before,candidate=after,oracle=ceiling,reduction_pct=100*(before-after)/before,
                      oracle_reduction_pct=100*(before-ceiling)/before,gap_closed_pct=100*(before-after)/(before-ceiling)),
        sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['port/upright_port.py','evidence/round12_before_port.py','evidence/round12_candidate_port.py']})
    (ROOT/'evidence/round12_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(full_gap=result['full_gap'],diagnostic=diagnostic,errors=errors),indent=2))

if __name__=='__main__':main()
