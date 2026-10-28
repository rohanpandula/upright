"""Experimental 3-line Manhattan consensus. Thresholds are CHOICE."""
import os

def _select_vps(seg_n,w,h,f0,seed):
    seg=seg_n*[w,h,w,h];d=max(w,h)
    centered=(seg-[w/2,h/2,w/2,h/2])/d
    lines=_homogeneous_lines(centered)
    length=np.linalg.norm(centered[:,2:]-centered[:,:2],axis=1)
    lines/=np.maximum(np.linalg.norm(lines[:,:2],axis=1)[:,None],_EPS)
    mids=(centered[:,:2]+centered[:,2:])/2
    power=float(os.environ.get('R13_POWER','2'))
    sigma=math.radians(float(os.environ.get('R13_SIGMA','.5')))
    prior=math.radians(float(os.environ.get('R13_PRIOR','25')))
    support=math.radians(float(os.environ.get('R13_SUPPORT','.5')))
    weights=(length/max(float(length.mean()),_EPS))**power
    rng=np.random.default_rng(seed);probability=length/length.sum()
    a=rng.choice(len(seg),2000,p=probability);b=rng.choice(len(seg),2000,p=probability)
    cand=np.cross(lines[a],lines[b]);cand=np.r_[cand,[[0.,1.,0.],[1.,0.,0.]]]
    norm=np.linalg.norm(cand,axis=1);cand=cand[norm>_EPS]/norm[norm>_EPS,None]
    rays=cand.copy();rays[:,:2]*=d/f0;rays=_normalize_rows(rays)
    def residual(v):
        return np.abs(np.einsum('ni,mi->nm',lines,v))/np.maximum(np.linalg.norm(v[None,:,:2]-mids[:,None,:]*v[None,:,2:],axis=2),_EPS)
    res=residual(cand)
    consensus=np.sum(weights[:,None]*np.exp(-.5*(res/sigma)**2),axis=0)
    angle=np.arccos(np.clip(abs(rays[:,1]),0,1))
    score=consensus*np.exp(-.5*(angle/prior)**2)
    modes=[]
    for idx in np.argsort(-score):
        if angle[idx]>math.radians(65):continue
        if any(abs(rays[idx]@rays[j])>math.cos(math.radians(2)) for j in modes):continue
        modes.append(idx)
        if len(modes)==24:break
    if not modes:return []
    y=rays[modes];y*=np.where(y[:,1:2]<0,-1.,1.)
    picks=rng.choice(len(seg),128,p=probability)
    normals=lines[picks]*[f0/d,f0/d,1]
    x=np.cross(y[:,None,:],normals[None,:,:]).reshape(-1,3)
    x=_normalize_rows(x);y=np.repeat(y,len(picks),axis=0)
    x*=np.where(np.cross(x,y)[:,2:]<0,-1.,1.)
    z=np.cross(x,y)
    good=(x[:,0]>.4)&(z[:,2]>.2)
    x,y,z=x[good],y[good],z[good]
    if not len(x):return []
    vx=x*[f0/d,f0/d,1];vy=y*[f0/d,f0/d,1]
    rv,rh=residual(vy),residual(vx)
    sv=np.sum(weights[:,None]*np.exp(-.5*(rv/sigma)**2),axis=0)
    sh=np.sum(weights[:,None]*np.exp(-.5*(rh/sigma)**2),axis=0)
    rotation_angle=np.arccos(np.clip((x[:,0]+y[:,1]+z[:,2]-1)/2,-1,1))
    score=sv*sh*np.exp(-.5*(rotation_angle/prior)**2)
    best=int(np.argmax(score))
    masks=[rv[:,best]<support,rh[:,best]<support]
    R=np.vstack([x[best],y[best],z[best]])
    # Joint rotation refinement preserves perpendicular family identities.
    K=_intrinsics_norm(f0,w,h);Ki=np.linalg.inv(K)
    mean=float(np.mean(length)*d)
    for _ in range(3):
        if min(map(np.count_nonzero,masks))<3:return []
        def loss(theta):
            H=K@_rotation(*theta)@Ki
            return np.r_[_family_terms(H,seg_n[masks[0]],np.array([1.,0.]),1,w,h,mean),
                         _family_terms(H,seg_n[masks[1]],np.array([0.,1.]),1,w,h,mean)]
        theta,_=_levenberg_marquardt(loss,_euler_xyz(R),np.full(3,-1.2),np.full(3,1.2),iters=30)
        R=_rotation(*theta)
        vs=(R[:2]*[f0/d,f0/d,1])[::-1]
        rr=residual(vs)
        masks=[rr[:,k]<support for k in range(2)]
    result=[]
    for k,mask in enumerate(masks):
        if np.count_nonzero(mask)<3:return []
        vp=K@R[1-k]
        result.append(dict(vp=_normalize(vp),inliers=np.flatnonzero(mask),weight=float(weights[mask].sum())))
    return result
