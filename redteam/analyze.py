import json, sys, numpy as np
from multiprocessing import Pool
from core import to_pix, fit, fit_D, decomp

RECS=[json.loads(l) for l in open("real.jsonl")]
W0=4000; H0=3000

def base_for(r, W, Hh, f):
    return dict(f0=f,f1=f,cx0=r["meta"]["UprightCenterNormX"]*W,cy0=r["meta"]["UprightCenterNormY"]*Hh,
                cx1=W/2,cy1=Hh/2,sk0=0,sk1=0,fx0=f,fy0=f,fx1=f,fy1=f)

def one(job):
    idx, slot = job
    r=RECS[idx]; W,Hh=r["w"],r["h"]
    Hn=np.array(r["tr"].get(str(slot)),float)
    if Hn is None: return None
    Hp=to_pix(Hn.reshape(3,3),W,Hh)
    if Hp is None: return None
    f35=r["meta"]["UprightFocalLength35mm"]/36.0*max(W,Hh)
    b=base_for(r,W,Hh,f35)
    out=dict(idx=idx,image=r["image"],slot=slot,f35=f35,W=W,H=Hh,
             ident=float(np.abs(Hn-np.eye(3).ravel()).max()))
    # report model
    v,p=fit(Hp,["f0","f1","cx0","cy0","cx1","cy1"],b,nstarts=5,seed=1)
    out["rep"]=v
    if p is not None and v<1e-6:
        d=decomp(Hp,p,["f0","f1","cx0","cy0","cx1","cy1"],b,False)
        R=None
        N=np.linalg.inv(np.array([[d["f1"],0,d["cx1"]],[0,d["f1"],d["cy1"]],[0,0,1]]))@Hp@np.array([[d["f0"],0,d["cx0"]],[0,d["f0"],d["cy0"]],[0,0,1]])
        M=N@N.T; s=np.sqrt(np.trace(M)/3); R=N/s
        out["pitch"]=float(np.degrees(np.arctan2(R[2,1],R[2,2])))
        out["roll"]=float(np.degrees(np.arctan2(R[1,0],R[0,0])))
        out["yaw"]=float(np.degrees(np.arcsin(np.clip(-R[2,0],-1,1))))
        out["f0"]=d["f0"]; out["f1"]=d["f1"]
        out["c0x"]=d["cx0"]; out["c0y"]=d["cy0"]; out["c1x"]=d["cx1"]; out["c1y"]=d["cy1"]
        out["detR"]=float(np.linalg.det(R)); out["ang"]=d["ang"]
    # shared f + both centres (5)
    v,p=fit(Hp,["f0","cx0","cy0","cx1","cy1"],b,shared=True,nstarts=5,seed=2)
    out["sharedF"]=v
    # single K general (5)
    v,p=fit(Hp,["fx0","fy0","sk0","cx0","cy0"],b,shared=True,nstarts=6,seed=3)
    out["singleKgen"]=v
    # single K free pp (3)
    v,p=fit(Hp,["f0","cx0","cy0"],b,shared=True,nstarts=5,seed=4)
    out["singleKpp"]=v
    # conventional shared f, centre fixed (1)
    v,p=fit(Hp,["f0"],b,shared=True,nstarts=3,seed=5)
    out["conv"]=v
    # D model (6)
    v,p=fit_D(Hp,b,nstarts=6,seed=6)
    out["Dmodel"]=v
    # f0 pinned at multiples, fit f1,c0,c1
    for m,tag in ((0.5,"pin0.5"),(1.0,"pin1"),(2.0,"pin2")):
        bb=dict(b); bb["f0"]=f35*m
        v,p=fit(Hp,["f1","cx0","cy0","cx1","cy1"],bb,shared=False,nstarts=4,seed=7)
        out[tag]=v
    return out

def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 300
    slot=int(sys.argv[2]) if len(sys.argv)>2 else 2
    rng=np.random.default_rng(0)
    # exclude exact identity for slot 2? keep for abstention stats
    idxs=list(range(len(RECS))); rng.shuffle(idxs); idxs=idxs[:n]
    with Pool(8) as pool:
        res=[x for x in pool.imap_unordered(one,[(i,slot) for i in idxs]) if x]
    json.dump(res,open(f"res_{n}_{slot}.json","w"))
    print("n",len(res))
    for k in ["rep","sharedF","singleKgen","singleKpp","conv","Dmodel","pin0.5","pin1","pin2"]:
        a=np.array([x[k] for x in res]); a=a[np.isfinite(a)]
        print("%-11s median %.3e  p90 %.3e  max %.3e  frac<1e-9 %.3f"%(k,np.median(a),np.percentile(a,90),a.max(),np.mean(a<1e-9)))

if __name__=="__main__": main()
