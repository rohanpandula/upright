import numpy as np, re, json, sys
d=open('x86.bin','rb').read()
TS,TE=0x1730,0x2c503a0
n=TE-TS
def build(los):
    cands=[]
    for lo,hi in los:
        for m in re.finditer(rb'[ -~]{4,}\x00', d[lo:hi]):
            cands.append((lo+m.start(), m.group()[:-1].decode()))
    return cands
def scan(los):
    cands=build(los)
    lab={a:s for a,s in cands}
    tset=np.array(sorted(lab),dtype=np.int64)
    hits=[]
    for ph in range(4):
        cnt=(n-ph)//4
        arr=np.frombuffer(d,dtype='<i4',count=cnt,offset=TS+ph).astype(np.int64)
        pos=TS+ph+4*np.arange(cnt,dtype=np.int64)
        res=pos+4+arr
        for i in np.nonzero(np.isin(res,tset))[0]:
            hits.append((int(pos[i]),int(res[i])))
    hits.sort()
    return hits,lab
if __name__=="__main__":
    los=eval(sys.argv[1])
    hits,lab=scan(los)
    for a,t in hits: print(hex(a),hex(t),lab[t])
