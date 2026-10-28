"""Artifact-level checks for Round 11 coordinate and paired-measurement seams."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evidence'))
from round11_probe import load_variant,up
from round11_analyze import load,BASE,ACTUAL,analyze

assert hashlib.sha256((ROOT/'port/upright_port.py').read_bytes()).hexdigest()=='f2a900d9e7640a5d47b38a1e61589ebbea316ace194aa0594e8c7badd6c52342'
cv2.setNumThreads(1)
sample={r['image_id']:r for r in json.loads((ROOT/'evidence/round11_sample.json').read_text())}
assert len(sample)==450
for path in (ROOT/'evidence/round11_supports').glob('*.npz'):
    with np.load(path) as z:
        assert np.isfinite(z['seg']).all(),path
        assert np.isfinite(z['Q']).all(),path
# Re-decoding and solving the raw control reproduces all 80 Round 10 maps/decisions.
raw=[p for p in load(ROOT/'evidence/round11_decode_40.jsonl') if p['variant']=='raw']
assert len(raw)==40
for p in raw:
    for r,b,a in zip(p['rows'],BASE[p['image_id']]['rows'],ACTUAL[p['image_id']]['rows']):
        assert r['accepted']==a['candidate_ok']
        assert r['fit']==b['candidate_ok']
        np.testing.assert_array_equal(r['H'],b['H'])
# Recover actual detector pixels from inverse-Adobe supports: catch Q, scale,
# homogeneous division and active-slot errors in the key bypass experiment.
for p in load(ROOT/'evidence/round11_adobe_all.jsonl')[:3]:
    r=sample[p['image_id']];gray,Q,_=load_variant(r,'adobe');h0,w0=gray.shape
    w,h=int(r['width']),int(r['height'])
    if r['orientation'] in ('BC','DA','AD','CB'):w,h=h,w
    expected=up.detect_segments(gray)*[w/w0,h/h0,w/w0,h/h0]
    z=np.load(ROOT/f'evidence/round11_supports/{r["image_id"]}_adobe.npz');seg=z['seg']
    J=Q@np.array(r['transform'][str(r['adobe']['mode'])]).reshape(3,3)@np.linalg.inv(Q)
    pts=seg.reshape(-1,2)/[w,h];pts=np.column_stack([pts,np.ones(len(pts))])@J.T
    got=(pts[:,:2]/pts[:,2:]*[w,h]).reshape(-1,4)
    np.testing.assert_allclose(got,expected,rtol=1e-11,atol=1e-8)
# No failure disappears from complete summaries.
preview=analyze(ROOT/'evidence/round11_preview_all.jsonl')['preview']
oracle=analyze(ROOT/'evidence/round11_oracle_all.jsonl')['oracle_labels']
assert preview['variant']['errors']==4 and oracle['variant']['errors']==26
for s in [preview,oracle]:
    for m in ['auto','full']:
        r=s['complete'][m];assert r['n']==387
        for v in ['candidate','returned']:
            x=r[v];assert x['win']+x['lose']+x['tie']+x['invalid']==387
assert oracle['complete']['full']['candidate']['error']['median']<1
print('PASS: production hash unchanged; 80 raw control maps and decisions exact; 3 Adobe backmap round trips; all 387 failures/fallbacks retained.')
