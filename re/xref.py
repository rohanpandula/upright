import capstone, struct, sys, json
IMG=open('/tmp/cr_x86.bin','rb').read()
TEXT_LO=0x1730; TEXT_HI=0x2c503a0
TARGETS=set()
for a in sys.argv[1:]:
    TARGETS.add(int(a,16))
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
md.detail=True
hits={t:[] for t in TARGETS}
pos=TEXT_LO
while pos<TEXT_HI:
    ok=False
    for ins in md.disasm(IMG[pos:pos+15],pos,count=1):
        ok=True
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:
                tgt=ins.address+ins.size+op.mem.disp
                if tgt in hits:
                    hits[tgt].append((ins.address,ins.mnemonic,ins.op_str))
        pos+=ins.size
    if not ok: pos+=1
for t,v in hits.items():
    print(hex(t),len(v))
    for x in v[:30]: print('   ',hex(x[0]),x[1],x[2])
