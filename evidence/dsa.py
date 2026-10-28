import sys,struct
from capstone import *
from capstone.x86 import *
d=open(sys.argv[1],'rb').read()
md=Cs(CS_ARCH_X86,CS_MODE_64); md.detail=True
lo=int(sys.argv[2],16); hi=int(sys.argv[3],16)
p=lo
def note(t):
    if t<0 or t+16>len(d): return ''
    b=d[t:t+16]
    out=[]
    for k in range(0,16,8):
        out.append('f64=%r'%struct.unpack('<d',b[k:k+8])[0])
    out.append('f32x2=%r'%(struct.unpack('<ff',b[:8]),))
    if 0x8ea5310<=t<0x8fea907:
        s=''.join(chr(c) if 32<=c<127 else '.' for c in b[:24])
        out.append('str=%r'%s)
    return '   ; '+' '.join(out)
while p<hi:
    n=0
    for ins in md.disasm(d[p:hi],p):
        ex=''
        for op in ins.operands:
            if op.type==X86_OP_MEM and op.mem.base==X86_REG_RIP:
                ex=note(ins.address+ins.size+op.mem.disp)
        print(hex(ins.address),ins.mnemonic,ins.op_str,ex); n+=1
    if n==0: p+=1
    else:
        p=ins.address+ins.size
        if p<=ins.address: p+=1
