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
    power=float(os.environ.get('R13_POWER','1'))
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
    residual=np.abs(np.einsum('ni,mi->nm',lines,cand))/np.maximum(np.linalg.norm(cand[None,:,:2]-mids[:,None,:]*cand[None,:,2:],axis=2),_EPS)
    consensus=np.sum(weights[:,None]*np.exp(-.5*(residual/sigma)**2),axis=0)
    # Keep distinct angular modes before testing pair compatibility.
    choices=[]
    for axis in (1,0):
        angle=np.arccos(np.clip(abs(rays[:,axis]),0,1))
        votes=weights[:,None]*np.exp(-.5*(residual/sigma)**2)
        coordinate=mids[:,1-axis]
        moment=np.sum(votes*coordinate[:,None],axis=0)/np.maximum(consensus,_EPS)
        variance=np.sum(votes*(coordinate[:,None]-moment)**2,axis=0)/np.maximum(consensus,_EPS)
        score=consensus*np.sqrt(variance+.0025)*np.exp(-.5*(angle/prior)**2)
        order=np.argsort(-score)
        modes=[]
        for idx in order:
            if angle[idx]>math.radians(65):continue
            if any(abs(rays[idx]@rays[j])>math.cos(math.radians(3)) for j in modes):continue
            modes.append(idx)
            if len(modes)==16:break
        choices.append([(idx,score[idx]) for idx in modes])
    frames=[]
    for vi,vs in choices[0]:
        for hi,hs in choices[1]:
            orth=abs(rays[vi]@rays[hi])
            score=vs*hs*math.exp(-.5*(orth/math.sin(math.radians(5)))**2)
            frames.append((score,vi,hi))
    if not frames:return []
    _,vi,hi=max(frames)
    N=np.array([[1/d,0,-w/(2*d)],[0,1/d,-h/(2*d)],[0,0,1.]])
    result=[]
    for best in (vi,hi):
        v=cand[best]
        mask=residual[:,best]<support
        if np.count_nonzero(mask)<3:return []
        # Refine with the same length weighting as the consensus score.
        for _ in range(5):
            denom=np.maximum(np.linalg.norm(v[:2]-mids*v[2],axis=1),_EPS)
            error=np.einsum('ni,i->n',lines,v)/denom
            wt=np.sqrt(weights[mask]*np.exp(-.5*(error[mask]/sigma)**2))/denom[mask]
            _,_,vt=np.linalg.svd(lines[mask]*wt[:,None],full_matrices=False)
            v=_normalize(vt[-1])
            error=np.abs(np.einsum('ni,i->n',lines,v))/np.maximum(np.linalg.norm(v[:2]-mids*v[2],axis=1),_EPS)
            mask=error<support
            if np.count_nonzero(mask)<3:return []
        vp=np.linalg.inv(N)@v
        result.append(dict(vp=_normalize(vp/[w,h,1]),inliers=np.flatnonzero(mask),weight=float(weights[mask].sum())))
    return result
