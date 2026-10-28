import numpy as np, sys
d=open('x86.bin','rb').read()
TS,TE=0x1730,0x2c503a0
n=TE-TS
for a in sys.argv[1:]:
    tgt=int(a,16); hits=[]
    for ph in range(4):
        cnt=(n-ph)//4
        rel=np.frombuffer(d,dtype='<i4',count=cnt,offset=TS+ph+1).astype(np.int64)
        pos=TS+ph+4*np.arange(cnt,dtype=np.int64)
        m=np.isin(pos+5+rel,tgt)
        hits+=[int(p) for p in pos[m]]
    print(hex(tgt),len(hits),[hex(x) for x in sorted(set(hits))[:20]])
