import json, sys, numpy as np
from multiprocessing import Pool
from core import to_pix, fit, fit_D, decomp

RECS=[json.loads(l) for l in open("real.jsonl")]

def base_for(r,W,Hh,f):
    return dict(f0=f,f1=f,cx0=r["meta"]["UprightCenterNormX"]*W,cy0=r["meta"]["UprightCenterNormY"]*Hh,
                cx1=W/2,cy1=Hh/2,sk0=0,sk1=0,fx0=f,fy0=f,fx1=f,fy1=f)

def one(job):
    idx,slot=job; r=RECS[idx]; W,Hh=r["w"],r["h"]
    Hn=np.array(r["tr"].get(str(slot)),float)
    if Hn is None: return None
    Hp=to_pix(Hn.reshape(3,3),W,Hh)
    if Hp is None: return None
    f35=r["meta"]["UprightFocalLength35mm"]/36.0*max(W,Hh)
    b=base_for(r,W,Hh,f35)
    out=dict(idx=idx,image=r["image"],slot=slot,f35=f35,W=W,H=Hh,ident=float(np.abs(Hn-np.eye(3).ravel()).max()))
    v,p=fit(Hp,["f0","f1","cx0","cy0","cx1","cy1"],b,nstarts=6,seed=1)
    out["rep"]=v
    if p is not None and v<1e-6:
        d=decomp(Hp,p,["f0","f1","cx0","cy0","cx1","cy1"],b,False)
        K0=np.array([[d["f0"],0,d["cx0"]],[0,d["f0"],d["cy0"]],[0,0,1]])
        K1=np.array([[d["f1"],0,d["cx1"]],[0,d["f1"],d["cy1"]],[0,0,1]])
        N=np.linalg.inv(K1)@Hp@K0; M=N@N.T; s=np.sqrt(np.trace(M)/3); R=N/s
        out.update(pitch=float(np.degrees(np.arctan2(R[2,1],R[2,2]))),
                   roll=float(np.degrees(np.arctan2(R[1,0],R[0,0]))),
                   yaw=float(np.degrees(np.arcsin(np.clip(-R[2,0],-1,1)))),
                   f0=d["f0"],f1=d["f1"],c0x=d["cx0"],c0y=d["cy0"],c1x=d["cx1"],c1y=d["cy1"],detR=float(np.linalg.det(R)))
    v,p=fit(Hp,["f0","cx0","cy0"],b,shared=True,nstarts=5,seed=4); out["singleKpp"]=v
    v,p=fit(Hp,["fx0","fy0","sk0","cx0","cy0"],b,shared=True,nstarts=6,seed=3); out["singleKgen"]=v
    v,p=fit_D(Hp,b,nstarts=6,seed=6); out["Dmodel"]=v
    for m,tag in ((0.25,"pin0.25"),(0.5,"pin0.5"),(1.0,"pin1"),(2.0,"pin2"),(4.0,"pin4")):
        bb=dict(b); bb["f0"]=f35*m
        v,p=fit(Hp,["f1","cx0","cy0","cx1","cy1"],bb,shared=False,nstarts=4,seed=7)
        out[tag]=v
    # K0 fully fixed to stored: only f1,c1 free
    bb=dict(b); bb["f0"]=f35
    v,p=fit(Hp,["f1","cx1","cy1"],bb,shared=False,nstarts=5,seed=8); out["storedK0"]=v
    return out

def main():
    n=int(sys.argv[1]); slot=int(sys.argv[2])
    rng=np.random.default_rng(1)
    idxs=list(range(len(RECS))); rng.shuffle(idxs); idxs=idxs[:n]
    with Pool(8) as pool:
        res=[x for x in pool.imap_unordered(one,[(i,slot) for i in idxs]) if x]
    json.dump(res,open(f"res2_{n}_{slot}.json","w"))
    print("n",len(res))
    for k in ["rep","singleKpp","singleKgen","Dmodel","pin0.25","pin0.5","pin1","pin2","pin4","storedK0"]:
        a=np.array([x[k] for x in res]); a=a[np.isfinite(a)]
        print("%-10s median %.3e p90 %.3e max %.3e frac<1e-9 %.3f"%(k,np.median(a),np.percentile(a,90),a.max(),np.mean(a<1e-9)))
if __name__=="__main__": main()
