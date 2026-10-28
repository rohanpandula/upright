from pathlib import Path
import json,sys,cv2,numpy as np
sys.path[:0]=['evidence','port']
from round9_probe import load_photo
import round13_candidate_port as up
rows=json.loads(Path('evidence/round10_sample.json').read_text());ids=[3362080,2120446,6736816,4583151,3986508,6644984]
canvas=np.zeros((2*400,3*600,3),np.uint8)
for k,iid in enumerate(ids):
 r=next(r for r in rows if r['image_id']==iid)
 gray,Q,_=load_photo(r);h0,w0=gray.shape
 sw,sh=r['width'],r['height'];w,h=(sh,sw) if r['orientation'] in ('BC','DA','AD','CB') else (sw,sh)
 z=np.load(f'evidence/round10_supports/{iid}.npz');s=z['seg']
 families=up._select_vps(s/[w,h,w,h],w,h,up.metadata_focal_px(r['meta'].get('UprightFocalLength35mm'),w,h),0)
 im=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
 for a,family in enumerate(families):
  for seg in s[family['inliers']]*[w0/w,h0/h,w0/w,h0/h]:
   cv2.line(im,tuple(seg[:2].astype(int)),tuple(seg[2:].astype(int)),[(0,255,100),(0,100,255)][a],2)
 scale=min(590/w0,360/h0);im=cv2.resize(im,None,fx=scale,fy=scale)
 y=(k//3)*400;x=(k%3)*600
 canvas[y+30:y+30+im.shape[0],x:x+im.shape[1]]=im
 cv2.putText(canvas,str(iid),(x,y+23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
cv2.imwrite('/tmp/upright-r13-families.jpg',canvas)
