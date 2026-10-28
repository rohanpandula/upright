"""Image-derived selector experiment. No target maps or filenames in selection."""
import itertools
import math
import numpy as np
import upright_port as up


def estimate_vanishing_points(segments,w,h,seed=0):
    seg=np.asarray(segments,float)
    if len(seg)<6:return []
    d=max(w,h);center=np.array([w/2,h/2,w/2,h/2]);s=(seg-center)/d
    lines=up._homogeneous_lines(s)
    lines/=np.maximum(np.linalg.norm(lines[:,:2],axis=1)[:,None],up._EPS)
    starts=np.c_[s[:,:2],np.ones(len(s))];mids=np.c_[(s[:,:2]+s[:,2:])/2,np.ones(len(s))]
    length=np.linalg.norm(seg[:,2:]-seg[:,:2],axis=1)
    weight=length/max(length.mean(),up._EPS)
    tau=max(up.TAU,.002*d)/d
    # Dual-space plane consensus. Long supports seed planes, all supports vote.
    rng=np.random.default_rng(seed)
    prob=np.sqrt(length);prob/=prob.sum()
    ia=rng.choice(len(s),up.RANSAC_ITERS,p=prob)
    ib=rng.choice(len(s),up.RANSAC_ITERS,p=prob)
    cand=np.cross(lines[ia],lines[ib])
    cand=np.r_[cand,np.c_[lines[:,1],-lines[:,0],np.zeros(len(lines))]]
    norms=np.linalg.norm(cand,axis=1);cand=cand[norms>up._EPS]/norms[norms>up._EPS,None]
    scored={}
    for v in cand:
        residual=up._vp_residual(starts,mids,v,tau)
        mask=residual<tau
        if mask.sum()<3:continue
        key=np.packbits(mask).tobytes()
        score=float(np.sum(weight*np.maximum(0,1-(residual/tau)**2)))
        if key not in scored or score>scored[key][0]:scored[key]=(score,v,mask)
    accepted=[];covered=np.zeros(len(s),bool)
    N=np.array([[1/d,0,-w/(2*d)],[0,1/d,-h/(2*d)],[0,0,1.]])
    Ni=np.linalg.inv(N)
    candidates=list(scored.values())
    while candidates and len(accepted)<9:
        gains=[float(weight[mask & ~covered].sum()) for _,_,mask in candidates]
        i=int(np.argmax(gains));score,v,mask=candidates.pop(i)
        if np.count_nonzero(mask & ~covered)<3 or gains[i]<=0:break
        fitted=up._refine_vp(seg[mask],w,h,Ni@v)
        vn=up._normalize(N@fitted)
        if any(abs(vn@up._normalize(N@a['vp']))>math.cos(math.radians(3)) for a in accepted):continue
        accepted.append(dict(vp=fitted,inliers=np.flatnonzero(mask),weight=score));covered|=mask
    if len(accepted)<2:return []
    for _ in range(3):
        res=np.stack([up._vp_residual(starts,mids,N@a['vp'],tau) for a in accepted],axis=1)
        best=np.argmin(res,axis=1);out=[]
        for k,a in enumerate(accepted):
            idx=np.flatnonzero((best==k)&(res[:,k]<tau))
            if len(idx)<3:continue
            fitted=up._refine_vp(seg[idx],w,h,a['vp'])
            out.append(dict(vp=fitted,inliers=idx,weight=float(weight[idx].sum())))
        accepted=out
        if len(accepted)<2:return []
    return sorted(accepted,key=lambda a:-a['weight'])


def classify(vps,seg_n,w=1,h=1,f0=None):
    if len(vps)<2:return None,[]
    f0=f0 or max(w,h)
    K=up._intrinsics_norm(f0,w,h);Ki=np.linalg.inv(K)
    rays=np.array([up._normalize(Ki@v['vp']) for v in vps])
    frames=[]
    for a,b in itertools.combinations(range(len(vps)),2):
        orth=abs(rays[a]@rays[b])
        if orth>math.sin(math.radians(15)):continue
        z=up._normalize(np.cross(rays[a],rays[b]))
        # Estimate the frame first. Its Y ray defines roll, not segment angles.
        axes=np.array([rays[a],rays[b],z]);vertical=int(np.argmax(abs(axes[:,1])))
        if vertical==2:continue
        vi=[a,b][vertical];hi=[b,a][vertical]
        if abs(rays[hi,0])<abs(z[0]):continue
        R=up._rotation_from_vps(vps[vi]['vp'],vps[hi]['vp'],Ki)
        if R is None or not np.isfinite(R).all():continue
        angles=up._euler_xyz(R)
        score=vps[vi]['weight']*vps[hi]['weight']*math.exp(-.5*(orth/.15)**2-.5*sum((angles/np.radians([20,20,30]))**2))
        # A third observed axis must agree with the frame; arbitrary unused VPs
        # never supply the optimizer's depth/horizon term.
        depth=None
        for k in range(len(vps)):
            if k in (vi,hi):continue
            match=abs(rays[k]@R[2])
            if match>math.cos(math.radians(5)):
                if depth is None or vps[k]['weight']>vps[depth]['weight']:depth=k
        if depth is not None:score*=1+vps[depth]['weight']/max(vps[vi]['weight']+vps[hi]['weight'],up._EPS)
        frames.append((float(score),vi,hi,depth,R))
    if not frames:return None,[]
    frames.sort(key=lambda f:-f[0]);best=frames[0]
    runner=next((f[0] for f in frames[1:] if np.linalg.norm(up._euler_xyz(f[4]@best[4].T))>math.radians(5)),0.)
    if not (runner/best[0]<=.75 and best[0]-runner>1.5):return None,[]
    v,h,dep=best[1:4]
    return vps[v],[vps[h]]+([vps[dep]] if dep is not None else [])
