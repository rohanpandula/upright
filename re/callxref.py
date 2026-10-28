import capstone, sys
IMG=open('/tmp/cr_x86.bin','rb').read()
LO=0x1730; HI=0x2c503a0
T={int(a,16) for a in sys.argv[1:]}
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64); md.detail=True
pos=LO; hits={t:[] for t in T}
while pos<HI:
    ok=False
    for ins in md.disasm(IMG[pos:pos+15],pos,count=1):
        ok=True
        if ins.mnemonic=='call' and ins.operands and ins.operands[0].type==capstone.x86.X86_OP_IMM:
            t=ins.operands[0].imm
            if t in hits: hits[t].append(ins.address)
        pos+=ins.size
    if not ok: pos+=1
for t,v in hits.items():
    print(hex(t), len(v), [hex(x) for x in v[:25]])
