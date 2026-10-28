import struct
P="/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/Frameworks/CameraRaw.lrtoolkit/Versions/A/CameraRaw"
FAT=open(P,'rb').read()
BASE_OFF=16384
SIZE=156736560
IMG=FAT[BASE_OFF:BASE_OFF+SIZE]
m,ct,cs,ft,ncmds,szcmds,flags,res=struct.unpack("<IiiIIIII",IMG[:32])
print("macho magic",hex(m),"cputype",ct,"ncmds",ncmds)
segs=[];secs=[]
off=32
for i in range(ncmds):
    cmd,cs_=struct.unpack("<II",IMG[off:off+8])
    if cmd==0x19:
        name=IMG[off+8:off+24].rstrip(b'\0').decode()
        vmaddr,vmsize,fileoff,filesize=struct.unpack("<QQQQ",IMG[off+24:off+56])
        maxprot,initprot,nsects,fl=struct.unpack("<iiII",IMG[off+56:off+72])
        segs.append((name,vmaddr,vmsize,fileoff,filesize))
        so=off+72
        for j in range(nsects):
            sname=IMG[so:so+16].rstrip(b'\0').decode()
            sg=IMG[so+16:so+32].rstrip(b'\0').decode()
            addr,size=struct.unpack("<QQ",IMG[so+32:so+48])
            fo,al,ro,nr,fl2,res2,res3=struct.unpack("<IIIIIII",IMG[so+48:so+76])
            secs.append((sg,sname,addr,size,fo,fl2))
            so+=80
    elif cmd==0x80000034:
        print("CHAINED FIXUPS at",off)
    elif cmd==0x22:
        print("DYLD_INFO")
    off+=cs_
for s in segs: print("SEG",s[0],hex(s[1]),hex(s[2]),hex(s[3]),hex(s[4]))
for s in secs: print("SEC",s[0],s[1],hex(s[2]),hex(s[3]),hex(s[4]),hex(s[5]))

def findptr(v):
    p=struct.pack("<Q",v); out=[]; s=0
    while True:
        i=IMG.find(p,s)
        if i<0: break
        out.append(i); s=i+1
    return out

def str_at(va):
    e=IMG.find(b'\0',va); return IMG[va:e].decode('latin1')
