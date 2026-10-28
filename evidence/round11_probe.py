"""Round 11 input/support interventions. Never writes catalog or changes solver policy."""
from pathlib import Path
import argparse
import concurrent.futures
import io
import json
import random
import re
import sqlite3
import sys
import time
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'port'),str(ROOT/'evidence'),str(ROOT)]
import upright_port as up
from benchmark_adobe import agreement
from round9_probe import load_photo,orient,json_default
from harvest_lr_transforms import parse_text
CACHE=ROOT/'evidence/round11_supports'
RAW={'.arw','.dng','.nef','.raf','.cr2','.3fr'}
PREV=Path('/Volumes/SSD/Lightroom/Lightroom Catalog Previews.lrdata')


def inventory():
    sample=[r for r in json.loads((ROOT/'evidence/round10_sample.json').read_text()) if r['cohort']!='abstain']
    cat=sqlite3.connect('file:/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat?mode=ro',uri=True)
    prev=sqlite3.connect(f'file:{PREV}/previews.db?mode=ro',uri=True)
    for r in sample:
        d=cat.execute('select digest,croppedWidth,text from Adobe_imageDevelopSettings where image=?',(r['image_id'],)).fetchone()
        p=prev.execute('select uuid,digest,orientation from ImageCacheEntry where imageId=?',(r['image_id'],)).fetchone()
        settings=parse_text(d[2]);mode=int(settings.get('PerspectiveUpright',0))
        manual=any(settings.get(k,default)!=default for k,default in [('PerspectiveVertical',0),('PerspectiveHorizontal',0),('PerspectiveRotate',0),('PerspectiveX',0),('PerspectiveY',0),('PerspectiveAspect',0),('PerspectiveScale',100)])
        eligible=p and p[1]==d[0] and d[1]=='uncropped' and not manual and mode in (1,2) and 'CropAngle' not in d[2]
        r['adobe']={'digest_match':bool(p and p[1]==d[0]),'uncropped':d[1]=='uncropped','manual':manual,'mode':mode,'eligible':bool(eligible),'lens_profile':bool(re.search(r'LensProfileEnable\s*=\s*1',d[2]))}
        if eligible:
            levels=prev.execute('select longDimension from PyramidLevel where uuid=? and digest=? order by longDimension desc',p[:2]).fetchall()
            for level, in levels:
                path=PREV/p[0][0]/p[0][:4]/f'{p[0]}-{p[1]}_{int(level)}'
                if path.is_file():r['adobe']['path']=str(path);break
            r['adobe']['eligible']='path' in r['adobe']
    (ROOT/'evidence/round11_sample.json').write_text(json.dumps(sample,indent=2)+'\n')
    print('inventory',len(sample),'eligible',sum(r['adobe']['eligible'] for r in sample),'full',sum(r['adobe']['eligible'] and r['adobe']['mode']==2 for r in sample),flush=True)


def gray_rgb(rgb,r):
    # Storage coordinates throughout. Catalog orientation is applied exactly once.
    scale=up.DEFAULT_MAX_SIDE/max(r['width'],r['height'])
    size=(round(r['width']*scale),round(r['height']*scale))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.resize(gray,size,interpolation=cv2.INTER_AREA)
    return orient(gray,r['orientation'])


def load_variant(r,variant):
    if variant=='raw':return load_photo(r)
    if variant.startswith('radial_'):
        gray,Q,info=load_photo(r);k=float(variant.split('_')[1]);h,w=gray.shape
        yy,xx=np.mgrid[:h,:w].astype(np.float32);cx=(w-1)/2;cy=(h-1)/2
        radius2=((xx-cx)**2+(yy-cy)**2)/(cx*cx+cy*cy)
        gray=cv2.remap(gray,cx+(xx-cx)*(1+k*radius2),cy+(yy-cy)*(1+k*radius2),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        info.update(radial_k=k,radial_normalization='unit radius at corners; source = center + output_offset*(1+k*r^2)')
        return gray,Q,info
    if variant=='adobe':
        with Image.open(r['adobe']['path']) as im:rgb=np.asarray(im.convert('RGB'))
        gray,Q=gray_rgb(rgb,r)
        return gray,Q,dict(decoder='Adobe digest-matched cached JPEG',shape=list(rgb.shape),**r['adobe'])
    import rawpy
    with rawpy.imread(r['path']) as raw:
        if variant=='preview':
            t=raw.extract_thumb()
            if t.format==rawpy.ThumbFormat.JPEG:
                with Image.open(io.BytesIO(t.data)) as im:
                    exif_orientation=im.getexif().get(274);rgb=np.asarray(im.convert('RGB'))
            else:rgb=t.data;exif_orientation=None
            info=dict(decoder='LibRaw embedded thumbnail',format=str(t.format),exif_orientation=exif_orientation)
        else:
            rgb=raw.postprocess(half_size=variant!='full_decode',user_flip=0,use_camera_wb=True,no_auto_bright=variant!='bright',output_bps=8)
            sz=raw.sizes;factor=rgb.shape[1]/sz.width
            if sz.crop_width>0 and sz.crop_height>0:
                x=round((sz.crop_left_margin-sz.left_margin)*factor);y=round((sz.crop_top_margin-sz.top_margin)*factor)
                cw,ch=round(sz.crop_width*factor),round(sz.crop_height*factor)
                if x>=0 and y>=0 and x+cw<=rgb.shape[1]+1 and y+ch<=rgb.shape[0]+1:rgb=rgb[y:y+ch,x:x+cw]
            info=dict(decoder=variant)
    info['shape']=list(rgb.shape);info['aspect_error']=(rgb.shape[1]/rgb.shape[0])/(r['width']/r['height'])-1
    if abs(info['aspect_error'])>.02:raise ValueError(f'unsupported preview aspect: {info}')
    if variant=='preview' and max(rgb.shape[:2])<up.DEFAULT_MAX_SIDE*.8:raise ValueError(f'preview too small: {info}')
    gray,Q=gray_rgb(rgb,r)
    return gray,Q,info


def support_stats(seg,gray):
    lengths=np.linalg.norm(seg[:,2:]-seg[:,:2],axis=1)
    return dict(count=len(seg),length_q=np.percentile(lengths,[10,50,90,99]).tolist() if len(seg) else [],length_total=float(lengths.sum()),long50=int((lengths>=50).sum()),long100=int((lengths>=100).sum()),mean=float(gray.mean()),std=float(gray.std()),dark_fraction=float((gray<16).mean()),clipped_fraction=float((gray>250).mean()),laplacian_std=float(cv2.Laplacian(gray,cv2.CV_32F).std()))


def supports(r,variant):
    CACHE.mkdir(exist_ok=True)
    target=CACHE/f'{r["image_id"]}_{variant}.npz'
    if target.exists():
        with np.load(target) as z:return z['seg'],z['Q'],json.loads(str(z['info']))
    gray,Q,info=load_variant(r,variant)
    seg=up.detect_segments(gray)
    info['stats']=support_stats(seg,gray)
    wh,ww=gray.shape
    w,h=int(r['width']),int(r['height'])
    if r['orientation'] in ('BC','DA','AD','CB'):w,h=h,w
    seg=seg*[w/ww,h/wh,w/ww,h/wh]
    if variant=='adobe':
        native=np.array(r['transform'][str(r['adobe']['mode'])]).reshape(3,3)
        J=Q@native@np.linalg.inv(Q)
        pts=seg.reshape(-1,2)/[w,h]
        pts=up._apply_h(np.linalg.inv(J),np.column_stack([pts,np.ones(len(pts))]))
        pts=pts[:,:2]/pts[:,2:]*[w,h]
        seg=pts.reshape(-1,4)
        info['backmap']='inverse active Adobe Upright, oracle diagnostic; lens-corrected source plane'
    np.savez_compressed(target,seg=seg,Q=Q,info=json.dumps(info))
    return seg,Q,info


def solve_rows(r,seg,Q):
    sw,sh=int(r['width']),int(r['height']);w,h=(sh,sw) if r['orientation'] in ('BC','DA','AD','CB') else (sw,sh)
    f=up.metadata_focal_px(r['meta'].get('UprightFocalLength35mm'),w,h)
    result=[]
    for mode,slot in [('auto','1'),('full','2')]:
        native=np.array(r['transform'][slot]).reshape(3,3)
        def score(H):
            try:return agreement(native,np.linalg.inv(Q)@H@Q,sw,sh)
            except (ValueError,np.linalg.LinAlgError) as exc:return dict(raw_rms_pct=None,error=str(exc))
        captured=[]
        original=up._alignment_quality
        def bypass(*args):
            q=original(*args);captured.append(q);return dict(q,passed=True)
        with patch.object(up,'_alignment_quality',side_effect=bypass):
            out=up.solve_ex(seg,w,h,f,mode,seed=0,framing=True,scaling=True)
        quality=captured[-1] if captured else None
        accepted=bool(out['ok'] and quality and quality['passed'])
        row=dict(mode=mode,fit=out['ok'],accepted=accepted,reason=out['reason'],quality=quality,identity=score(np.eye(3)),candidate=score(out['H']) if out['ok'] else None,returned=score(out['H'] if accepted else np.eye(3)),rotation=out['rotation'],damping=out['damping'],source_focal_px=out['source_focal_px'],calibration=out['calibration'],H=out['H'].tolist())
        result.append(row)
    return result


def measure(job):
    r,variant=job;cv2.setNumThreads(1);start=time.time()
    try:
        seg,Q,info=supports(r,variant)
        return dict(image_id=r['image_id'],path=r['path'],variant=variant,info=info,rows=solve_rows(r,seg,Q),seconds=time.time()-start)
    except Exception as exc:return dict(image_id=r['image_id'],path=r['path'],variant=variant,error=repr(exc),seconds=time.time()-start)


def oracle_measure(r):
    """Oracle axis labels on the SAME raw detections; not a deployable method."""
    cv2.setNumThreads(1)
    try:
        z=np.load(ROOT/f'evidence/round10_supports/{r["image_id"]}.npz');seg=z['seg'];Q=z['Q']
        sw,sh=int(r['width']),int(r['height']);w,h=(sh,sw) if r['orientation'] in ('BC','DA','AD','CB') else (sw,sh)
        native=np.array(r['transform']['2']).reshape(3,3);J=Q@native@np.linalg.inv(Q)
        pts=seg.reshape(-1,2)/[w,h];p=up._apply_h(J,np.column_stack([pts,np.ones(len(pts))]));p=p[:,:2]/p[:,2:]*[w,h]
        d=np.diff(p.reshape(-1,2,2),axis=1)[:,0];length=np.linalg.norm(d,axis=1)
        a=np.degrees(np.arctan2(abs(d[:,1]),abs(d[:,0])))
        indices=[np.flatnonzero((a>=89)&(length>0)),np.flatnonzero((a<=1)&(length>0))]
        if min(map(len,indices))<3:
            return dict(image_id=r['image_id'],variant='oracle_labels',error='insufficient oracle axis supports',family_counts=list(map(len,indices)))
        def vps(*args,**kwargs):
            return [dict(vp=up._family_vp(seg[idx]/[w,h,w,h])*[w,h,1],inliers=idx) for idx in indices]
        with patch.object(up,'estimate_vanishing_points',side_effect=vps),patch.object(up,'_classify',side_effect=lambda v,s:(v[0],v[1:])):
            rows=solve_rows(r,seg,Q)
        return dict(image_id=r['image_id'],path=r['path'],variant='oracle_labels',info={'family_counts':list(map(len,indices))},rows=rows)
    except Exception as exc:return dict(image_id=r['image_id'],variant='oracle_labels',error=repr(exc))


def no_depth_measure(r):
    """Keep selected supports/calibration unchanged; remove only depth horizon term."""
    cv2.setNumThreads(1)
    z=np.load(ROOT/f'evidence/round10_supports/{r["image_id"]}.npz');seg=z['seg'];Q=z['Q']
    original=up._energy_residuals
    def without_depth(theta,K0,K0i,vert,horiz,vp_h2,*args):
        return original(theta,K0,K0i,vert,horiz,None,*args)
    with patch.object(up,'_energy_residuals',side_effect=without_depth):rows=solve_rows(r,seg,Q)
    return dict(image_id=r['image_id'],variant='no_depth',rows=rows)


def run(stage,limit,workers):
    sample=json.loads((ROOT/'evidence/round11_sample.json').read_text())
    if stage=='adobe':sample=[r for r in sample if r['adobe']['eligible']];variants=['adobe']
    else:
        sample=[r for r in sample if Path(r['path']).suffix.lower() in RAW]
        variants={'preview':['preview'],'decode':['raw','preview','full_decode','bright'],'radial':['radial_-0.05','radial_0.05','radial_-0.10','radial_0.10'],'oracle':['oracle_labels'],'no_depth':['no_depth']}[stage]
    random.Random(11011).shuffle(sample)
    if limit:sample=sample[:limit]
    # A frozen random prefix supports a paired pilot; extra full-mode Adobe cases are kept in a separate run.
    jobs=[(r,v) for r in sample for v in variants]
    target=ROOT/f'evidence/round11_{stage}_{limit or "all"}.jsonl'
    with target.open('w') as fh,concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results=pool.map(oracle_measure,sample) if stage=='oracle' else pool.map(no_depth_measure,sample) if stage=='no_depth' else pool.map(measure,jobs)
        for i,out in enumerate(results,1):
            fh.write(json.dumps(out,default=json_default)+'\n');fh.flush()
            if i%10==0 or i==len(jobs):print(stage,i,'/',len(jobs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','adobe','preview','decode','radial','oracle','no_depth']);p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage=='inventory':inventory()
    else:run(a.stage,a.limit,a.workers)
