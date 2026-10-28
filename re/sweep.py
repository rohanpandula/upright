import struct,capstone,pickle,collections
src=open('/Users/rohan/Projects/trichroma/research/upright/re/mo.py').read()
exec(src.split('for i in range(ncmds)')[0])
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64); md.detail=True
md.skipdata=True
A,B=0x22a0000,0x2310000
DISP=collections.defaultdict(list)
RIP=collections.defaultdict(list)
for ins in md.disasm(IMG[A:B],A):
    try: ops=ins.operands
    except Exception: continue
    for o in ops:
        if o.type==capstone.x86.X86_OP_MEM:
            if o.mem.base==capstone.x86.X86_REG_RIP:
                RIP[ins.address+ins.size+o.mem.disp].append(ins.address)
            elif o.mem.disp and o.mem.base in (capstone.x86.X86_REG_RDI,capstone.x86.X86_REG_RBX,capstone.x86.X86_REG_R12,capstone.x86.X86_REG_R13,capstone.x86.X86_REG_R14,capstone.x86.X86_REG_R15,capstone.x86.X86_REG_RAX,capstone.x86.X86_REG_RCX,capstone.x86.X86_REG_RSI,capstone.x86.X86_REG_RDX):
                DISP[o.mem.disp].append((ins.address,ins.mnemonic,ins.op_str))
pickle.dump((dict(DISP),dict(RIP)),open('/Users/rohan/Projects/trichroma/research/upright/re/sweep.pkl','wb'))
print("disp keys",len(DISP),"rip",len(RIP))
