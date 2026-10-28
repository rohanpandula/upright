"""Round 9 frozen-policy holdout and real-photo benchmark. No production changes."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'port'), str(ROOT)]
import upright_port as up
from benchmark_adobe import agreement, mapped_points
from make_heldout_images import CASES, render

C = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
POLICY_SHA = '60f842ea1d03356f4c411b00f3e0e0802dc0a0a8f20a86a468268979fa7b14ad'


def quad(H):
    q = C @ H.T
    if not np.isfinite(q).all() or q[:, 2].min() <= 0:
        raise ValueError('invalid/horizon quad')
    return q[:, :2]/q[:, 2:]


def signed_area(q):
    return float(np.sum(q[:, 0]*np.roll(q[:, 1], -1)-q[:, 1]*np.roll(q[:, 0], -1))/2)


def intersection_area(q):
    p = list(q)
    for axis, bound, sign in ((0, 0., 1), (0, 1., -1), (1, 0., 1), (1, 1., -1)):
        out = []
        for a, b in zip(p, p[1:]+p[:1]):
            ia, ib = sign*(a[axis]-bound) >= 0, sign*(b[axis]-bound) >= 0
            if ia:
                out.append(a)
            if ia != ib:
                out.append(a+(bound-a[axis])/(b[axis]-a[axis])*(b-a))
        p = out
        if not p:
            return 0.
    return abs(signed_area(np.array(p)))


def zoom(H, s):
    return np.array([[s, 0, (1-s)/2], [0, s, (1-s)/2], [0, 0, 1.]]) @ H


def geometry(H):
    B = up.frame_output(H)
    # scale_output(B) frames once internally, too. Repeating that exact operation
    # matters at equality: one ulp at 1.10 can change a discontinuous branch.
    p = quad(up.frame_output(B))-.5
    e = np.roll(p, -1, axis=0)-p
    n = np.column_stack([e[:, 1], -e[:, 0]])
    d = np.sum(n*p, axis=1)
    if np.any(d <= 0):
        raise ValueError('non-enclosing/degenerate quad')
    cover = float(np.max(np.sum(abs(n), axis=1)/(2*d)))
    inner = float(.5/np.max(abs(np.sort(p, axis=0)[1:3])))
    scale = cover if cover <= 1.10*inner else inner
    actual = up.scale_output(B)
    matrix_error = float(np.max(abs(actual-zoom(B,scale))))
    q = quad(actual)
    return dict(cover=cover, inner=inner, ratio=cover/inner, scale=scale,
                branch='cover' if cover <= 1.10*inner else 'inner',
                finite=bool(np.isfinite(actual).all()), area=signed_area(q),
                overlap=intersection_area(q), policy_matrix_error=matrix_error,
                bounds=[q.min(0).tolist(), q.max(0).tolist()])


def holdout():
    rows = []
    for case in CASES:
        image, _, _, truth = render(case)
        h, w = image.shape
        segments = up.detect_segments(image)
        f = up.focal_px_from_35mm(case['focal_mm'], w, h)
        # Ground-truth quad is a policy-only control; never enters solve_ex.
        control = geometry(truth)
        for seed in range(5):
            for mode in ('auto', 'full', 'level', 'vertical'):
                out = up.solve_ex(segments, w, h, f, mode, seed=seed,
                                  framing=True, scaling=True)
                rows.append(dict(scene=case['name'], case=case, seed=seed, mode=mode,
                                 ok=out['ok'], reason=out['reason'], segments=len(segments),
                                 H=out['H'].tolist(),
                                 geometry=geometry(out.get('H_unscaled', out['H'])),
                                 truth_geometry=control))
        print('heldout', case['name'], flush=True)
    return rows


def from_quad(q):
    # Double-precision 8-equation homography, with no OpenCV float32 quantization.
    rows, rhs = [], []
    for (x, y, _), (u, v) in zip(C, q):
        rows += [[x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]]
        rhs += [u,v]
    return np.r_[np.linalg.solve(rows, rhs), 1.].reshape(3, 3)


def sweep():
    rng = np.random.default_rng(9009)
    rows, outside, invalid, failures = [], [], [], []
    # Four shape families, both axes/signs, three aspect stretches, three sizes.
    for family in ('shear', 'rotation', 'trapezoid', 'projective'):
        for a in np.linspace(0, 1, 1001):
            if family == 'shear':
                q = C[:, :2] @ np.array([[1., 0], [.5*a, 1.]])
            elif family == 'rotation':
                t = np.pi*a/4
                q = (C[:, :2]-.5) @ np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])+.5
            elif family == 'trapezoid':
                q = np.array([[-.5-.5*a,-.5],[.5+.5*a,-.5],[.5-.35*a,.5],[-.5+.35*a,.5]])+.5
            else:
                H = np.array([[1., .2*a, 0], [.1*a, 1., 0], [.45*a, -.25*a, 1.]])
                q = quad(H)
            for swap in (False, True):
                for sign in (-1., 1.):
                    for aspect in (.5, 1., 2.):
                        for size in (.1, 1., 10.):
                            p = (q-.5)*[aspect*size, size]
                            if swap:
                                p = p[:, ::-1]
                            p[:, 0] *= sign
                            if signed_area(p) < 0:
                                p = p[[1,0,3,2]]
                            H = from_quad(p+.5)
                            try:
                                g = geometry(H)
                            except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
                                failures.append(dict(family=family, parameter=float(a), swap=swap,
                                    sign=sign, aspect=aspect, size=size, error=repr(exc), H=H.tolist()))
                                continue
                            if 1-1e-10 <= g['ratio'] <= 1.5+1e-10:
                                B = up.frame_output(H)
                                # Force both candidate branches, including the unused branch.
                                checks = []
                                for label in ('cover', 'inner'):
                                    J = zoom(B, g[label]); z = quad(J)
                                    checks.append(dict(branch=label, finite=bool(np.isfinite(J).all()),
                                        inversion=signed_area(z) <= 0, off_frame=intersection_area(z) <= 0,
                                        overlap=intersection_area(z)))
                                rows.append(dict(family=family, parameter=float(a), swap=swap, sign=sign,
                                    aspect=aspect, size=size, geometry=g, checks=checks))
    # Unconditioned stress controls: include horizon, reflection, singular and non-enclosing cases.
    for i in range(10000):
        H = rng.normal(size=(3,3)); H[2,2] = 1
        try:
            q = quad(H)
            g = geometry(H)
            outside.append(dict(ratio=g['ratio'], finite=g['finite'], area=g['area'], overlap=g['overlap']))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
            invalid.append(str(exc))
    sensitivity = []
    for ratio in np.round(np.arange(1.05,1.151,.01), 2):
        H = np.array([[1., ratio-1, -(ratio-1)/2], [0,1.,0], [0,0,1.]])
        g = geometry(H)
        cases = []
        for threshold in np.round(np.arange(1.05,1.151,.01), 2):
            s = g['cover'] if g['cover'] <= threshold*g['inner'] else g['inner']
            cases.append(dict(threshold=float(threshold), scale=s,
                rms_vs_frozen=agreement(up.scale_output(H), zoom(H,s), 1200,800)['raw_rms_pct']))
        sensitivity.append(dict(target_ratio=float(ratio), geometry=g, thresholds=cases,
            branch_gap_pct=agreement(zoom(H,g['cover']),zoom(H,g['inner']),1200,800)['raw_rms_pct']))
    eps = []
    for ratio in (1.10-1e-7, 1.10-1e-10, 1.10, 1.10+1e-10, 1.10+1e-7):
        H = np.array([[1.,ratio-1,-(ratio-1)/2],[0,1.,0],[0,0,1.]])
        eps.append(dict(target=ratio, geometry=geometry(H), H=up.scale_output(H).tolist()))
    assert rows and len(sensitivity) == 11
    return dict(rows=rows, family_failures=failures, stress_accepted=outside, stress_rejections=invalid,
                sensitivity=sensitivity, boundary=eps)


def orient(a, code):
    # Q maps sensor-normalized coordinates to the catalog's viewing orientation.
    cases = {
      'AB': (lambda x:x, [[1,0,0],[0,1,0],[0,0,1]]),
      'BA': (lambda x:x[:,::-1], [[-1,0,1],[0,1,0],[0,0,1]]),
      'CD': (lambda x:x[::-1,::-1], [[-1,0,1],[0,-1,1],[0,0,1]]),
      'DC': (lambda x:x[::-1], [[1,0,0],[0,-1,1],[0,0,1]]),
      'AD': (lambda x:np.swapaxes(x,0,1), [[0,1,0],[1,0,0],[0,0,1]]),
      'BC': (lambda x:np.rot90(x,-1), [[0,-1,1],[1,0,0],[0,0,1]]),
      'CB': (lambda x:np.swapaxes(x,0,1)[::-1,::-1], [[0,-1,1],[-1,0,1],[0,0,1]]),
      'DA': (lambda x:np.rot90(x,1), [[0,1,0],[-1,0,1],[0,0,1]])}
    fun, Q = cases[code]
    return np.ascontiguousarray(fun(a)), np.array(Q,dtype=float)


def load_photo(r):
    path = r['path']; suffix = Path(path).suffix.lower()
    info = {}
    if suffix in ('.arw','.dng','.nef','.raf','.cr2','.3fr'):
        import rawpy
        with rawpy.imread(path) as raw:
            sz = raw.sizes
            rgb = raw.postprocess(half_size=True, user_flip=0, use_camera_wb=True,
                                  no_auto_bright=True, output_bps=8)
            # LibRaw DefaultCropOrigin is relative to full RAW, postprocess to visible RAW.
            factor = rgb.shape[1]/sz.width
            if sz.crop_width > 0 and sz.crop_height > 0:
                x = round((sz.crop_left_margin-sz.left_margin)*factor)
                y = round((sz.crop_top_margin-sz.top_margin)*factor)
                cw, ch = round(sz.crop_width*factor), round(sz.crop_height*factor)
                if x >= 0 and y >= 0 and x+cw <= rgb.shape[1]+1 and y+ch <= rgb.shape[0]+1:
                    rgb = rgb[y:y+ch,x:x+cw]
                    info['raw_crop'] = [x,y,cw,ch]
            info['raw_visible'] = [sz.width,sz.height]
            info['decoder'] = 'rawpy half_size cameraWB no_auto_bright user_flip=0'
    elif suffix in ('.tif','.tiff'):
        # Pillow/libtiff silently applies TIFF Orientation while decoding. Read
        # storage pixels so Lightroom orientation is applied exactly once.
        import tifffile
        rgb = tifffile.imread(path)
        info['tiff_dtype'] = str(rgb.dtype)
        if rgb.dtype == np.uint16:
            rgb = np.rint(rgb/257).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:,:,None],3,axis=2)
        if rgb.shape[0] in (3,4) and rgb.shape[-1] not in (3,4):
            rgb = rgb.transpose(1,2,0)
        rgb = np.ascontiguousarray(rgb[:,:,:3])
        info['decoder'] = 'tifffile storage RGB; uint16 /257; no orientation'
    else:
        with Image.open(path) as im:
            info['raster_exif_orientation'] = im.getexif().get(274)
            rgb = np.asarray(im.convert('RGB'))
            info['decoder'] = 'Pillow RGB, no EXIF transpose'
    info['decoded_shape'] = list(rgb.shape)
    info['aspect_error'] = float((rgb.shape[1]/rgb.shape[0])/(r['width']/r['height'])-1)
    # Detect at the default 1200-pixel working size; measure_photo rescales
    # supports to original catalog pixels before solving and applying the gate.
    scale = up.DEFAULT_MAX_SIDE/max(r['width'],r['height'])
    sw, sh = round(r['width']*scale), round(r['height']*scale)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray,(sw,sh),interpolation=cv2.INTER_AREA)
    gray, Q = orient(gray, r['orientation'])
    info['working_shape'] = list(gray.shape)
    info['luminance_mean'] = float(gray.mean())
    info['luminance_std'] = float(gray.std())
    return gray, Q, info


def json_default(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v,np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def measure_photo(r):
    start = time.time()
    cv2.setNumThreads(1)
    try:
        gray, Q, decode = load_photo(r)
    except Exception as exc:
        return dict(image_id=r['image_id'], path=r['path'], decode_error=repr(exc))
    wh,ww = gray.shape
    w,h = int(r['width']),int(r['height'])
    if r['orientation'] in ('BC','DA','AD','CB'):
        w,h = h,w
    f = up.metadata_focal_px(r['meta'].get('UprightFocalLength35mm'),w,h)
    seg = up.detect_segments(gray)*[w/ww,h/wh,w/ww,h/wh]
    rows = []
    for mode,slot in (('auto','1'),('full','2')):
        baseline = up.solve_ex(seg,w,h,f,mode,seed=0)
        out = up.solve_ex(seg,w,h,f,mode,seed=0,framing=True,scaling=True)
        H = out['H']; original = baseline['H']
        framed = out.get('H_unscaled', original)
        if out['ok']:
            np.testing.assert_array_equal(original,out['H_unframed'])
        # Catalog maps are in sensor axes; compare both outputs in that same gauge.
        native = np.array(r['transform'][slot]).reshape(3,3)
        versions = {k:np.linalg.inv(Q)@v@Q for k,v in
                    (('without',original),('framing',framed),('both',H))}
        metrics = {}
        for key,value in versions.items():
            try:
                metrics[key] = agreement(native,value,r['width'],r['height'])
            except ValueError as exc:
                metrics[key] = dict(error=str(exc),raw_rms_pct=None)
        row = dict(image_id=r['image_id'],mode=mode,orientation=r['orientation'],
            ok=out['ok'],reason=out['reason'],without_ok=baseline['ok'],without_reason=baseline['reason'],
            segments=len(seg),decode=decode,
            rotation=out['rotation'],damping=out['damping'],quality=out['quality'],
            source_focal_px=out['source_focal_px'],input_focal_mm=r['meta'].get('UprightFocalLength35mm'),
            source_focal_mm=up.focal_35mm_from_px(out['source_focal_px'],w,h) if out['source_focal_px'] else None,
            calibration=out['calibration'], metrics=metrics,H={k:v.tolist() for k,v in versions.items()},
            native_H=native.tolist(), geometry=geometry(framed) if out['ok'] else None,
            native_perspective=float(np.linalg.norm(native[2,:2])),
            native_identity_distance=agreement(native,np.eye(3),r['width'],r['height'])['raw_rms_pct'],
            path=r['path'])
        rows.append(row)
    return dict(image_id=r['image_id'],rows=rows,seconds=time.time()-start)


def corpus(workers, limit):
    sample = json.loads((ROOT/'evidence/round9_sample.json').read_text())[:limit]
    path = ROOT/('evidence/round9_corpus.jsonl' if limit==300 else 'evidence/round9_pilot.jsonl')
    with path.open('w') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i,result in enumerate(pool.map(measure_photo,sample),1):
                f.write(json.dumps(result,default=json_default)+'\n');f.flush()
                summary = result.get('decode_error') or [(r['mode'],r['ok'],r['metrics']['both']['raw_rms_pct']) for r in result['rows']]
                print(i,result['image_id'],summary,flush=True)


def repair_tiffs(workers):
    # Correct the decoder seam on the SAME frozen sample; retain the first run.
    sample=json.loads((ROOT/'evidence/round9_sample.json').read_text())
    subset=[r for r in sample if Path(r['path']).suffix.lower() in ('.tif','.tiff')]
    path=ROOT/'evidence/round9_corpus.jsonl'
    prior=path.read_text()
    (ROOT/'evidence/round9_corpus_before_tiff_fix.jsonl').write_text(prior)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        replacements={r['image_id']:r for r in pool.map(measure_photo,subset)}
    with (ROOT/'evidence/round9_tiff_recheck.jsonl').open('w') as f:
        for r in replacements.values():
            f.write(json.dumps(r,default=json_default)+'\n')
    rows=[json.loads(l) for l in prior.splitlines()]
    assert len(rows)==300 and len(replacements)==33
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(replacements.get(r['image_id'],r),default=json_default)+'\n')


def native_control():
    """Frozen Round 8 fixed-source gauge; native rotation is scoring-only here."""
    from scipy.linalg import rq
    rows=[]
    for r in json.loads((ROOT/'evidence/round9_sample.json').read_text()):
        w,h=r['width'],r['height']
        f=up.focal_px_from_35mm(r['meta']['UprightFocalLength35mm'],w,h)
        K=up._intrinsics_norm(f,w,h);P=np.diag([w,h,1.])
        for mode,slot in (('auto','1'),('full','2')):
            H=np.array(r['transform'][slot]).reshape(3,3);H/=H[2,2]
            D,R=rq(P@H@K)
            signs=np.diag(np.where(np.diag(D)<0,-1.,1.))
            D=D@signs;R=signs@R;D/=D[2,2]
            s=float(np.sqrt(D[0,0]*D[1,1])/f)
            D[:2,:2]/=s;D[:2,2]=[w/2,h/2]
            B=np.linalg.inv(P)@D@R@np.linalg.inv(K);B/=B[2,2]
            B=up.frame_output(B);g=geometry(B)
            pred=up.scale_output(B)
            candidates={k:agreement(H,zoom(B,g[k]),w,h)['raw_rms_pct'] for k in ('cover','inner')}
            rows.append(dict(image_id=r['image_id'],mode=mode,native_scale=s,geometry=g,
                raw=agreement(H,pred,w,h)['raw_rms_pct'],candidates=candidates,
                native_framing_error=agreement(H,up.frame_output(H),w,h)['raw_rms_pct'],
                reconstructed_native_scale_raw=agreement(H,zoom(B,s),w,h)['raw_rms_pct'],
                path=r['path']))
    return rows


def boundary_controls():
    rows=[]
    for shear in (.1,-.1):
        H=np.array([[1.,shear,-shear/2],[0,1.,0],[0,0,1.]])
        original=up.scale_output(H)
        for label,A in (('direct',H),('preframed',up.frame_output(H)),
            ('translated_1e-15',np.array([[1,0,1e-15],[0,1,1e-15],[0,0,1.]])@H)):
            B=up.scale_output(A)
            rows.append(dict(shear=shear,variant=label,scale=float((B@np.linalg.inv(A))[0,0]),
                raw_vs_direct=agreement(original,B,1200,800)['raw_rms_pct'],H=A.tolist(),output=B.tolist()))
    diamond=np.array([[1.,-1,.5],[1,1,-.5],[0,0,1.]])
    try:
        up.scale_output(diamond)
        failure=None
    except ZeroDivisionError as exc:
        failure=repr(exc)
    assert failure is not None
    shifted=np.array([[1.,0,3],[0,1,0],[0,0,1.]])
    return dict(equality=rows,diamond=dict(H=diamond.tolist(),error=failure),
        unframed_shifted_overlap=intersection_area(quad(up.scale_output(shifted))),
        framed_shifted_overlap=intersection_area(quad(up.scale_output(up.frame_output(shifted)))))


def direct_jpegs():
    rows=[]
    for r in json.loads((ROOT/'evidence/round9_sample.json').read_text()):
        if Path(r['path']).suffix.lower()!='.jpg':
            continue
        with Image.open(r['path']) as im:
            rgb=np.asarray(im.convert('RGB'))
        gray,_=orient(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),r['orientation'])
        h,w=gray.shape
        f=up.metadata_focal_px(r['meta']['UprightFocalLength35mm'],w,h)
        for mode in ('auto','full'):
            out=up.solve_ex(image=gray,focal_px=f,mode=mode,framing=True,scaling=True)
            rows.append(dict(image_id=r['image_id'],mode=mode,ok=out['ok'],reason=out['reason'],H=out['H']))
    assert len(rows)==24 and not any(r['ok'] for r in rows)
    assert all(np.array_equal(r['H'],np.eye(3)) for r in rows)
    return rows


def self_check():
    assert hashlib.sha256((ROOT/'port/upright_port.py').read_bytes()).hexdigest()==POLICY_SHA
    assert geometry(np.eye(3))['scale']==1
    assert abs(intersection_area(C[:,:2])-1)<1e-12
    test = np.array([[1,.2,-.1],[0,1,0],[0,0,1.]])
    assert abs(geometry(test)['ratio']-1.2)<1e-12
    np.testing.assert_allclose(from_quad(quad(test)),test,atol=1e-14)
    a = np.arange(6).reshape(2,3)
    for code in ('AB','BA','CD','DC','AD','BC','CB','DA'):
        b,Q = orient(a,code)
        # Pixel centres obey the normalized coordinate mapping including mirrors.
        for y in range(2):
            for x in range(3):
                q = Q@[(x+.5)/3,(y+.5)/2,1]
                assert b[int(q[1]*b.shape[0]),int(q[0]*b.shape[1])]==a[y,x]


def freeze_sample():
    import collections
    import random
    import sqlite3
    from harvest_lr_transforms import QUERY,parse_text
    counts=collections.Counter();available=[]
    with sqlite3.connect('file:/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat?mode=ro',uri=True) as con:
        con.execute('PRAGMA query_only=ON')
        for image,text,base,ext,folder,root,w,h,ori in con.execute(QUERY):
            p=parse_text(text);tr=p.pop('transforms');counts['with_auto_key']+=1
            if any(k not in tr or np.max(abs(np.array(tr[k]).reshape(3,3)-np.eye(3)))<1e-6 for k in (1,2)):
                continue
            counts['both_nonidentity']+=1
            path=os.path.join(root,folder,f'{base}.{ext}')
            if not Path(path).is_file():
                counts['missing']+=1
                continue
            counts['exists']+=1
            available.append(dict(image_id=image,path=path,width=w,height=h,orientation=ori,
                transform={str(k):v for k,v in tr.items()},meta=p))
    real=[r for r in available if '/research/upright/' not in r['path'] and Path(r['path']).suffix.lower() not in ('.mp4','.mov')]
    unique={}
    for r in real:
        unique.setdefault(r['path'],r)
    sample=random.Random(9009).sample(sorted(unique.values(),key=lambda r:r['image_id']),300)
    (ROOT/'evidence/round9_sample.json').write_text(json.dumps(sample,indent=2)+'\n')
    (ROOT/'evidence/round9_sampling.json').write_text(json.dumps(dict(counts=counts,
        available_real_rows=len(real),available_real_unique=len(unique),seed=9009),indent=2)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('sample','heldout','sweep','corpus','repair-tiffs','native','direct_jpeg','check'))
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--limit',type=int,default=300)
    args=ap.parse_args();self_check()
    if args.stage=='sample':
        freeze_sample()
    elif args.stage=='corpus':
        corpus(args.workers,args.limit)
    elif args.stage=='repair-tiffs':
        repair_tiffs(args.workers)
    elif args.stage!='check':
        result={'heldout':holdout,'sweep':sweep,'native':native_control,'direct_jpeg':direct_jpegs}[args.stage]()
        (ROOT/f'evidence/round9_{args.stage}.json').write_text(json.dumps(result,default=json_default)+'\n')
