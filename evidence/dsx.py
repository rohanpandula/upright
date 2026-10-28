import sys
from capstone import *
d=open(sys.argv[1],'rb').read()
md=Cs(CS_ARCH_X86,CS_MODE_64)
lo=int(sys.argv[2],16); hi=int(sys.argv[3],16)
p=lo
while p<hi:
    n=0
    for ins in md.disasm(d[p:hi],p):
        print(hex(ins.address),ins.mnemonic,ins.op_str); n+=1
    if n==0: p+=1
    else:
        p=ins.address+ins.size
        if p<=ins.address: p+=1
