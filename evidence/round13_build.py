from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]
source=(root/'port/upright_port.py').read_text()
start=source.index('    vps = estimate_vanishing_points(',source.index('def _base_solve'))
end=source.index('    # Lock the RANSAC',start)
source=source[:start]+'''    vps = _select_vps(seg_n, w, h, f0, seed)
    if len(vps) < 2:
        return None
    vertical, horizontal = vps[0], vps[1:]
    K0 = _intrinsics_norm(f0, w, h)
    K0i = np.linalg.inv(K0)

'''+source[end:]
source+='\n'+(root/'evidence/round13_selector.py').read_text()
(root/'evidence/round13_candidate_port.py').write_text(source)
