"""Check denominators, finite scores, implementation scope and final restoration."""
from pathlib import Path
import ast
import hashlib
import json
import re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence'


def load(name):return json.loads((E/name).read_text())


def main():
    before=(E/'round12_before_port.py').read_bytes();shipped=(ROOT/'port/upright_port.py').read_bytes()
    assert before==shipped
    assert hashlib.sha256(shipped).hexdigest()=='f2a900d9e7640a5d47b38a1e61589ebbea316ace194aa0594e8c7badd6c52342'
    def parsed(path):
        t=ast.parse(path.read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)},[ast.dump(n,include_attributes=False) for n in t.body if not isinstance(n,ast.FunctionDef)]
    a,ag=parsed(E/'round12_before_port.py');b,bg=parsed(E/'round12_candidate_port.py')
    assert ag==bg
    assert [k for k in a if a[k]!=b[k]]==['estimate_vanishing_points','_classify','_base_solve']
    assert b.keys()-a.keys()=={'_manhattan_margin'}
    rows=[json.loads(x) for x in (E/'round12_final_candidate_all.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['image_id'] for r in rows})==679
    assert [r['image_id'] for r in rows if 'error' in r]==[2116098]
    for p in rows:
        if 'error' in p:continue
        assert len(p['rows'])==2
        for r in p['rows']:
            assert np.isfinite(r['identity']['raw_rms_pct']) and np.isfinite(r['returned']['raw_rms_pct'])
            if r['fit']:assert np.isfinite(r['candidate']['raw_rms_pct']) and np.isfinite(r['H']).all()
            assert r['accepted']==bool(r['fit'] and r['quality'] and r['quality']['passed'])
    summary=load('round12_summary.json')
    assert summary['groups']['raw387']['candidate']['full']['n']==387
    assert summary['groups']['corrected450']['candidate']['full']['n']==450
    assert summary['groups']['abstention_decoded']['candidate']['full']['n']==228
    gap=summary['full_gap'];assert gap['gap_closed_pct']<0
    assert np.isclose(gap['gap_closed_pct'],100*(gap['before']-gap['candidate'])/(gap['before']-gap['oracle']))
    for backend in ['candidate','final_candidate','shipped']:
        assert len(load(f'round12_{backend}_sweep.json'))==640
        assert len(load(f'round12_{backend}_heldout.json'))==280
        assert len(load(f'round12_{backend}_adobe.json'))==160
        q=load(f'round12_{backend}_quality.json')
        trials=[t for s in q['scenes'].values() for t in s['trials']]
        assert len(trials)==1100 and all(t['ok'] and t['gate_pass'] for t in trials)
        assert max(max(t['returned_axis_deg']) for t in trials)<.1
    for name in ['round12_final_candidate_pytest.txt','round12_shipped_pytest.txt']:
        assert re.search(r'105 passed', (E/name).read_text())
    assert (E/'round12.txt').is_file()
    print('PASS: exact production restoration; candidate only changes selection; 679/678/450/387/228 denominators;')
    print('finite scored outputs; unchanged captured quality verdict; negative oracle-gap recovery;')
    print('all three synthetic validation sets complete; 105 tests pass for final candidate and production.')

if __name__=='__main__':main()
