"""Current production vs the corpus pitch curve, same scenes/seeds/focal protocols."""
import importlib.util, json, math, sys
sys.path.insert(0, 'port'); sys.path.insert(0, '.'); sys.path.insert(0, 'evidence')
import numpy as np, cv2
import upright_port as up
from benchmark_adobe import agreement
from validate_vs_adobe import CASES, SLOTS, parse_adobe

spec = importlib.util.spec_from_file_location('before', '/tmp/port_before_auto.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)

rows = []
for name, case in CASES.items():
    adobe = parse_adobe(f'adobe_truth/img_{case["text"]}.lua')
    text = open(f'adobe_truth/img_{case["text"]}.lua').read()
    import re
    meta = {k: float(re.search(k + r'\s*=\s*([-\d.eE]+)', text)[1])
            for k in ('UprightFocalLength35mm',)}
    img = cv2.imread(f'lrtest/in/{name}.tif', 0)
    h, w = img.shape
    seg = up.detect_segments(img)
    for prior, focal in (('scene_focal', 1600.),
                         ('stored_focal', up.focal_px_from_35mm(
                             meta['UprightFocalLength35mm'], w, h))):
        for variant, mod in (('before', before), ('after', up)):
            for seed in range(5):
                out = mod.solve_ex(seg, w, h, focal, 'auto', seed=seed)
                ag = agreement(adobe[SLOTS['auto']], out['H'], w, h)
                rows.append(dict(scene=name, prior=prior, variant=variant, seed=seed,
                                 raw=ag['raw_rms_pct'], shape=ag['shape_rms_pct'],
                                 damp=out['damping'][0] if out['damping'] else None,
                                 ok=out['ok']))
    print('done', name, flush=True)

for prior in ('scene_focal', 'stored_focal'):
    print(f"\n[{prior}] Auto only, 5 seeds, raw/shape percent of source diagonal")
    print(f"  {'scene':10s} {'before raw':>10} {'after raw':>10} {'d':>8} "
          f"{'before shp':>10} {'after shp':>10} {'d':>8} {'b.damp':>7} {'a.damp':>7}")
    agg = []
    for name in CASES:
        b = [r for r in rows if r['scene'] == name and r['prior'] == prior and r['variant'] == 'before']
        a = [r for r in rows if r['scene'] == name and r['prior'] == prior and r['variant'] == 'after']
        br, bs = np.mean([r['raw'] for r in b]), np.mean([r['shape'] for r in b])
        ar, as_ = np.mean([r['raw'] for r in a]), np.mean([r['shape'] for r in a])
        bd, ad = np.mean([r['damp'] for r in b]), np.mean([r['damp'] for r in a])
        agg.append((br, bs, ar, as_))
        print(f"  {name:10s} {br:10.4f} {ar:10.4f} {ar-br:+8.4f} "
              f"{bs:10.4f} {as_:10.4f} {as_-bs:+8.4f} {bd:7.4f} {ad:7.4f}")
    m = np.mean(agg, axis=0)
    print(f"  {'MEAN':10s} {m[0]:10.4f} {m[2]:10.4f} {m[2]-m[0]:+8.4f} "
          f"{m[1]:10.4f} {m[3]:10.4f} {m[3]-m[1]:+8.4f}")
json.dump(rows, open('/tmp/auto_corpus_verify.json', 'w'), indent=1)
