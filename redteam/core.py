import numpy as np
from scipy.optimize import least_squares

def Kmat(fx, fy, sk, cx, cy):
    return np.array([[fx, sk, cx],[0.0, fy, cy],[0.0,0.0,1.0]])

def to_pix(Hn, W, Hh):
    S = np.diag([1.0/W, 1.0/Hh, 1.0])
    Hp = np.linalg.inv(S) @ Hn @ S
    if abs(Hp[2,2]) < 1e-14: return None
    return Hp / Hp[2,2]

def _d(p, free, base):
    d = dict(base)
    for i,n in enumerate(free):
        d[n] = float(np.exp(np.clip(p[i], -25, 25))) if n.startswith("f") else float(p[i])
    return d

def Ks(d, free, shared):
    f=d["f0"]
    fx = d["fx0"] if "fx0" in free else f
    fy = d["fy0"] if "fy0" in free else fx
    sk = d["sk0"]; cx = d["cx0"]; cy = d["cy0"]
    K0 = Kmat(fx,fy,sk,cx,cy)
    if shared:
        return K0, K0
    f1=d["f1"]
    fx1 = d["fx1"] if "fx1" in free else f1
    fy1 = d["fy1"] if "fy1" in free else fx1
    return K0, Kmat(fx1,fy1,d["sk1"],d["cx1"],d["cy1"])

def resid(p, H, free, base, shared):
    d=_d(p,free,base); K0,K1=Ks(d,free,shared)
    N=np.linalg.inv(K1)@H@K0
    M=N@N.T; s2=np.trace(M)/3.0
    return ((M-s2*np.eye(3))/s2).ravel()

def fit(H, free, base, shared=False, nstarts=4, seed=0, jitter=0.3):
    rng=np.random.default_rng(seed)
    x0=np.array([np.log(max(base[n],1e-9)) if n.startswith("f") else base[n] for n in free])
    lo=np.array([np.log(1e-3) if n.startswith("f") else -2e5 for n in free])
    hi=np.array([np.log(1e9) if n.startswith("f") else 2e5 for n in free])
    best=(np.inf,None)
    for k in range(nstarts):
        x=np.clip(x0.copy(),lo,hi)
        if k>0:
            for i,n in enumerate(free):
                x[i]+= rng.normal(0,1)*(jitter if n.startswith("f") else (0.05*abs(base[n])+1.0))
            x=np.clip(x,lo,hi)
        try:
            r=least_squares(resid,x,args=(H,free,base,shared),method="trf",bounds=(lo,hi),
                            max_nfev=6000,xtol=1e-15,ftol=1e-15,gtol=1e-15)
        except Exception:
            continue
        v=float(np.abs(r.fun).max())
        if v<best[0]: best=(v,r.x)
    return best

def decomp(H, p, free, base, shared):
    d=_d(p,free,base); K0,K1=Ks(d,free,shared)
    N=np.linalg.inv(K1)@H@K0
    M=N@N.T; s=np.sqrt(np.trace(M)/3.0); R=N/s
    ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    return dict(f0=K0[0,0],f1=K1[0,0],cx0=K0[0,2],cy0=K0[1,2],cx1=K1[0,2],cy1=K1[1,2],
                s=s,ang=ang,det=np.linalg.det(R),resid=float(np.abs(M-s*s*np.eye(3)).max()/(s*s)))

# ---- single-camera + output-anisotropy control (6 intrinsic params) ----
def resid_D(p, H, base):
    # params: fx0,fy0,sk0,cx0,cy0,dx  (K1 = diag(dx,1,1) K0)
    d=dict(base)
    d["fx0"]=float(np.exp(np.clip(p[0],-25,25))); d["fy0"]=float(np.exp(np.clip(p[1],-25,25)))
    d["sk0"]=p[2]; d["cx0"]=p[3]; d["cy0"]=p[4]; dx=float(np.exp(np.clip(p[5],-25,25)))
    K0=Kmat(d["fx0"],d["fy0"],d["sk0"],d["cx0"],d["cy0"])
    D=np.diag([dx,1.0,1.0]); K1=D@K0
    N=np.linalg.inv(K1)@H@K0
    M=N@N.T; s2=np.trace(M)/3.0
    return ((M-s2*np.eye(3))/s2).ravel()

def fit_D(H, base, nstarts=6, seed=0):
    rng=np.random.default_rng(seed)
    x0=np.array([np.log(base["fx0"]),np.log(base["fy0"]),0.0,base["cx0"],base["cy0"],0.0])
    lo=np.array([np.log(1e-2),np.log(1e-2),-1e3,-2e5,-2e5,np.log(1e-2)])
    hi=np.array([np.log(1e7),np.log(1e7),1e3,2e5,2e5,np.log(1e4)])
    best=(np.inf,None)
    for k in range(nstarts):
        x=np.clip(x0.copy(),lo,hi)
        if k>0:
            for i in range(6): x[i]+=rng.normal(0,1)*(0.3 if i in (0,1,5) else 1.0)
            x=np.clip(x,lo,hi)
        try:
            r=least_squares(resid_D,x,args=(H,base),method="trf",bounds=(lo,hi),max_nfev=6000,xtol=1e-15,ftol=1e-15,gtol=1e-15)
        except Exception: continue
        v=float(np.abs(r.fun).max())
        if v<best[0]: best=(v,r.x)
    return best
