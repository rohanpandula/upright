"""Experimental angular consensus. All thresholds are CHOICE, not native facts."""
import os

def _select_vps(seg_n,w,h,f0,seed):
    seg=seg_n*[w,h,w,h]
    d=max(w,h)
    centered=(seg-[w/2,h/2,w/2,h/2])/d
    lines=_homogeneous_lines(centered)
    length=np.linalg.norm(centered[:,2:]-centered[:,:2],axis=1)
    lines/=np.maximum(np.linalg.norm(lines[:,:2],axis=1)[:,None],_EPS)
    mids=(centered[:,:2]+centered[:,2:])/2
    power=float(os.environ.get('R13_POWER','2'))
    sigma=math.radians(float(os.environ.get('R13_SIGMA','.5')))
    prior=math.radians(float(os.environ.get('R13_PRIOR','25')))
    support=math.radians(float(os.environ.get('R13_SUPPORT','1')))
    weights=(length/max(float(length.mean()),_EPS))**power
    rng=np.random.default_rng(seed)
    probability=length/length.sum()
    a=rng.choice(len(seg),2000,p=probability);b=rng.choice(len(seg),2000,p=probability)
    cand=np.cross(lines[a],lines[b])
    cand=np.r_[cand,[[0.,1.,0.],[1.,0.,0.]]]
    norm=np.linalg.norm(cand,axis=1);cand=cand[norm>_EPS]/norm[norm>_EPS,None]
    rays=cand.copy();rays[:,:2]*=d/f0;rays=_normalize_rows(rays)
    residual=np.abs(lines@cand.T)/np.maximum(np.linalg.norm(cand[None,:,:2]-mids[:,None,:]*cand[None,:,2:],axis=2),_EPS)
    consensus=np.sum(weights[:,None]*np.exp(-.5*(residual/sigma)**2),axis=0)
    result=[]
    for axis in (1,0):
        angle=np.arccos(np.clip(abs(rays[:,axis]),0,1))
        score=consensus*np.exp(-.5*(angle/prior)**2)
        order=np.argsort(-score)
        # Fit only the winning angular mode, not every line with a nearby endpoint.
        best=int(order[0]);v=cand[best]
        mask=residual[:,best]<support
        if np.count_nonzero(mask)<3:return []
        N=np.array([[1/d,0,-w/(2*d)],[0,1/d,-h/(2*d)],[0,0,1.]])
        for _ in range(3):
            vp=_refine_vp(seg[mask],w,h,np.linalg.inv(N)@v)
            v=_normalize(N@vp)
            error=np.abs(lines@v)/np.maximum(np.linalg.norm(v[:2]-mids*v[2],axis=1),_EPS)
            mask=error<support
            if np.count_nonzero(mask)<3:return []
        result.append(dict(vp=_normalize(vp/[w,h,1]),inliers=np.flatnonzero(mask),weight=float(weights[mask].sum())))
    return result
