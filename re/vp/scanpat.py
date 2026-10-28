import capstone,json,bisect,sys
d=open('x86.bin','rb').read()
fs=json.load(open('funcstarts.json'))
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
pats=sys.argv[1].split(',')
lo=int(sys.argv[2],16); hi=int(sys.argv[3],16)
i0=bisect.bisect_left(fs,lo); i1=bisect.bisect_left(fs,hi)
for j in range(i0,min(i1,len(fs)-1)):
    s,e=fs[j],fs[j+1]
    off=s
    while off<e:
        got=False
        for i in md.disasm(d[off:e],off):
            got=True
            for p in pats:
                if p in i.op_str:
                    print(f"{i.address:08x} fn={s:08x} {i.mnemonic} {i.op_str}")
            off=i.address+i.size
        if not got: break
