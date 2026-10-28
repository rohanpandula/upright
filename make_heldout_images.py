"""MIT. Independent NumPy ray/plane renderer for the Round 2 held-out set.

No PIL drawing, fixture geometry, port camera helpers or detector output.
Analytic material sampling gives soft grout edges, panel shading and grain.
The fixed 12-camera matrix plus single-wall and near-frontal controls is
specified before benchmarking. Changing cases after seeing scores invalidates
this holdout. Poses are degrees (pitch up, yaw right, roll).
"""
from pathlib import Path
import argparse
import json
import numpy as np

# Every aspect/focal combination; poses span the requested ranges.
CASES = [
    dict(name=f'H{i+1:02d}', w=w, h=h, focal_mm=f, pitch=p, yaw=y, roll=r,
         single_wall=False, seed=810+i)
    for i, (w,h,f,p,y,r) in enumerate([
        (960,640,24,5,0,0), (960,640,35,15,8,5),
        (960,640,50,28,17,10), (960,640,85,40,25,15),
        (960,720,24,35,20,12), (960,720,35,22,12,3),
        (960,720,50,10,4,8), (960,720,85,32,22,14),
        (1024,576,24,18,25,15), (1024,576,35,40,6,6),
        (1024,576,50,25,15,0), (1024,576,85,5,2,10)])]
CASES += [dict(name='H13_single_wall', w=960,h=720,focal_mm=35,
               pitch=23,yaw=13,roll=7,single_wall=True,seed=822),
          dict(name='H14_near_frontal', w=1024,h=576,focal_mm=50,
               pitch=5,yaw=.25,roll=.5,single_wall=False,seed=823)]


def camera(case):
    p,y,r = np.radians([-case['pitch'],case['yaw'],case['roll']])
    cx,sx,cy,sy,cz,sz = np.cos(p),np.sin(p),np.cos(y),np.sin(y),np.cos(r),np.sin(r)
    R = (np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @
         np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @
         np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]))
    w,h=case['w'],case['h']
    f=case['focal_mm']*np.hypot(w,h)/np.hypot(36,24)
    K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1]])
    return R,K


def clip_segment(a,b,w,h):
    """Liang-Barsky clipping in continuous pixel coordinates."""
    d=b-a; lo,hi=0.,1.
    for q,t in zip((-d[0],d[0],-d[1],d[1]),(a[0],w-1-a[0],a[1],h-1-a[1])):
        if abs(q)<1e-12:
            if t<0:return None
        elif q<0:lo=max(lo,t/q)
        else:hi=min(hi,t/q)
    return None if lo>hi else np.r_[a+lo*d,a+hi*d]


def world_lines(single_wall):
    lines=[]
    for x in np.arange(-25, 6.01, 1.35):
        lines.append(([(x,-20,14),(x,30,14)],0,'back'))
    for y in np.arange(-20,30.01,1.1):
        lines.append(([(-25,y,14),(6,y,14)],1,'back'))
    if not single_wall:
        for z in np.arange(2,14.01,1.35):
            lines.append(([(6,-20,z),(6,30,z)],0,'side'))
        for y in np.arange(-20,30.01,1.1):
            lines.append(([(6,y,2),(6,y,14)],2,'side'))
    return lines


def render(case):
    R,K=camera(case); w,h=case['w'],case['h']; f=K[0,0]
    yy,xx=np.mgrid[:h,:w].astype(float)
    rays=np.einsum('...j,ij->...i',
        np.stack([(xx-w/2)/f,(yy-h/2)/f,np.ones_like(xx)],axis=-1),R)
    back=14/rays[:,:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        side=6/rays[:,:,0]
    side_z=side*rays[:,:,2]
    use_side=(side>0)&(side<back)&(side_z>=2)&(side_z<=14)&(~np.full((h,w),case['single_wall']))
    distance=np.where(use_side,side,back)
    points=rays*distance[:,:,None]
    u=np.where(use_side,points[:,:,2]-2,points[:,:,0]+25)
    v=points[:,:,1]+20
    # Analytic softened rectangular grooves, with varying panel brightness.
    du=np.abs((u+.675)%1.35-.675)
    dv=np.abs((v+.55)%1.1-.55)
    grooves=np.maximum(.5-.5*np.tanh((du-.032)/.009),
                       .5-.5*np.tanh((dv-.026)/.012))
    panels=9*np.sin(np.floor(u/1.35)*1.71+np.floor(v/1.1)*.83)
    texture=2.2*np.sin(17*u+5*v)+1.6*np.cos(7*u-19*v)
    image=198-28*use_side+panels+texture-115*grooves
    image+=np.random.default_rng(case['seed']).normal(0,1.8,(h,w))
    image=np.clip(image,0,255).astype(np.uint8)
    truth=[]; kind=[]
    # Analytical projected material centerlines, clipped for visibility.
    for world,axis,plane in world_lines(case['single_wall']):
        if plane=='back' and case['single_wall']:
            # Single wall continues beyond the corner used by room scenes.
            world=np.array(world,dtype=float)
            if axis==1:world[1,0]=25
        projected=np.array(world) @ R @ K.T
        # Clip in camera depth before perspective division: at steep pitches
        # one endpoint is behind the camera although the line is visible.
        if np.all(projected[:,2]<=.01):continue
        for i in (0,1):
            if projected[i,2]<.01:
                j=1-i
                t=(.01-projected[i,2])/(projected[j,2]-projected[i,2])
                projected[i]+=t*(projected[j]-projected[i])
        pixels=projected[:,:2]/projected[:,2:]
        segment=clip_segment(*pixels,w,h)
        if segment is not None and np.linalg.norm(segment[2:]-segment[:2])>=40:
            truth.append(segment);kind.append(axis)
    N=np.diag([1/w,1/h,1])
    H=N@K@R@np.linalg.inv(K)@np.linalg.inv(N)
    return image,np.array(truth).reshape(-1,4),np.array(kind),H/H[2,2]


if __name__=='__main__':
    import cv2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'heldout')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for case in CASES:
        image,truth,kind,H=render(case)
        cv2.imwrite(str(args.output/(case['name']+'.png')),image)
        np.savez_compressed(args.output/(case['name']+'.npz'),segments=truth,kind=kind,H=H)
    (args.output/'manifest.json').write_text(json.dumps(CASES,indent=2)+'\n')
