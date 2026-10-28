import capstone,struct,sys,json,bisect
d=open('x86.bin','rb').read()
fs=json.load(open('funcstarts.json'))
def fname(a):
    i=bisect.bisect_right(fs,a)-1
    return fs[i]
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64); md.detail=True
s=int(sys.argv[1],16); e=int(sys.argv[2],16)
for i in md.disasm(d[s:e],s):
    extra=""
    for op in i.operands:
        if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:
            tgt=i.address+i.size+op.mem.disp
            b=d[tgt:tgt+8]
            if len(b)==8:
                dv=struct.unpack('<d',b)[0]
                fv=struct.unpack('<f',b[:4])[0]
                extra+=f"  ; [{tgt:#x}] d={dv!r} f={fv!r} u32={struct.unpack('<I',b[:4])[0]:#x}"
                if 0x8e00000<tgt<0x9000000:
                    st=d[tgt:tgt+70].split(b'\0')[0]
                    extra+=f" str={st.decode('latin1')!r}"
    if i.mnemonic=='call' and i.operands[0].type==capstone.x86.X86_OP_IMM:
        t=i.operands[0].imm
        extra+=f"  -> fn {fname(t):#x}" if 0x1730<t<0x2c503a0 else "  -> ext"
    print(f"{i.address:08x} {i.bytes.hex():<22} {i.mnemonic} {i.op_str}{extra}")
