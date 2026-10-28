"""Report-metric check on the non-Manhattan / mild families, before vs after."""
import importlib.util, sys
sys.path.insert(0, 'port'); sys.path.insert(0, '.'); sys.path.insert(0, 'evidence')
import numpy as np, cv2
import upright_port as up
from nm_analyze import sim_residual
from round5_limits import ideal_fraction
import make_test_images as M

spec = importlib.util.spec_from_file_location('before', '/tmp/port_before_auto.py')
before = importlib.util.module_from_spec(spec); spec.loader.exec_module(before)
grid = np.array([[(u, v) for u in np.linspace(60, M.W-60, 9)]
                 for v in np.linspace(60, M.H-60, 7)]).reshape(-1, 2)
CASES = [('NM_single_wall', 'nm'), ('NM_railing', 'nm'), ('NM_curves', 'nm'),
         ('NM_three_plane', 'nm'), ('CT_manhattan2', 'nm2')]
pose = [M.PITCH15, M.YAW6, 0.]
print(f"{'scene':16s} {'before':>9} {'after':>9} {'delta':>9}")
for name, folder in CASES:
    img = cv2.imread(f'lrtest/{folder}/{name}.tif', 0)
    h, w = img.shape
    seg = up.detect_segments(img)
    out = {}
    for tag, mod in (('before', before), ('after', up)):
        vals = []
        for seed in range(5):
            r = mod.solve_ex(seg, w, h, 1600., 'auto', seed=seed)
            vals.append(ideal_fraction(r['H'], pose) if r['ok'] else float('nan'))
        out[tag] = float(np.mean(vals))
    print(f"{name:16s} {out['before']:9.4f} {out['after']:9.4f} {out['after']-out['before']:+9.4f}")
