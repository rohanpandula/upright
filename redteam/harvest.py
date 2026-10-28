import sqlite3, re, json, sys
db="/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat"
con=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
cur=con.cursor()
RE_TR=re.compile(r'UprightTransform_(\d)\s*=\s*"([^"]*)"')
RE_NUM=re.compile(r'(\w+)\s*=\s*(-?[\d.]+)')
KEYS=["UprightFocalLength35mm","UprightCenterNormX","UprightCenterNormY","UprightFocalMode","UprightCenterMode","PerspectiveUpright"]
q="""SELECT d.image,d.text,i.fileWidth,i.fileHeight,i.orientation
FROM Adobe_imageDevelopSettings d JOIN Adobe_images i ON i.id_local=d.image
WHERE d.text LIKE '%UprightTransform_1%' AND i.fileWidth>0 AND i.fileHeight>0"""
n=0
out=open("real.jsonl","w")
for image,text,fw,fh,orient in cur.execute(q):
    tr={}
    for k,v in RE_TR.findall(text):
        try:
            nums=[float(x) for x in v.split(",")]
        except ValueError: continue
        if len(nums)==9: tr[k]=nums
    if "1" not in tr: continue
    meta={}
    for k,v in RE_NUM.findall(text):
        if k in KEYS: meta[k]=float(v)
    if "UprightFocalLength35mm" not in meta: continue
    out.write(json.dumps({"image":image,"w":fw,"h":fh,"orient":orient,"tr":tr,"meta":meta})+"\n")
    n+=1
print("wrote",n)
