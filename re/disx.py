import struct,capstone
src=open('/Users/rohan/Projects/trichroma/research/upright/re/mo.py').read()
exec(src.split('for i in range(ncmds)')[0])
md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64); md.detail=True
T0,T1=0x1730,0x1730+0x2c4ec70
def ann(ins):
    out=[]
    for o in ins.operands:
        if o.type==capstone.x86.X86_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:
            t=ins.address+ins.size+o.mem.disp
            if T0<=t<T1: out.append(f"<code {t:#x}>")
            else:
                try:
                    f=struct.unpack("<f",IMG[t:t+4])[0]; d=struct.unpack("<d",IMG[t:t+8])[0]
                    s=IMG[t:t+40].split(b'\0')[0]
                    txt=s.decode('latin1') if all(32<=c<127 for c in s) and len(s)>2 else ''
                    out.append(f"<{t:#x} f={f:.6g} d={d:.6g} '{txt}'>")
                except Exception: out.append(f"<{t:#x}>")
    return " ".join(out)
def dis(va,n=0x200,label=""):
    print(f"--- {va:#x} {label}")
    for ins in md.disasm(IMG[va:va+n],va):
        print(f"{ins.address:#08x}  {ins.mnemonic:<8}{ins.op_str}   {ann(ins)}")
