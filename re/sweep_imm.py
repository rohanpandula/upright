import capstone, collections
IMG=open('/tmp/cr_x86.bin','rb').read()
LO=0x1730; HI=0x2c503a0
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
md.detail=True
found=collections.defaultdict(list)
pos=LO
while pos<HI:
    ok=False
    for ins in md.disasm(IMG[pos:pos+15],pos,count=1):
        ok=True
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_IMM:
                v=op.imm & 0xffffffff
                if 0x08000000 <= v <= 0x0AFFFFFF and v!=0:
                    found[v].append(ins.address)
        pos+=ins.size
    if not ok: pos+=1
for v in sorted(found):
    print(hex(v), len(found[v]), [hex(x) for x in found[v][:8]])
