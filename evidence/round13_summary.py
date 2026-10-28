from pathlib import Path
import sys,json
import round11_analyze as a
for path in sys.argv[1:]:
 p=a.load(path);ids=[r['image_id'] for r in p if 'rows' in r]
 for name,data in [('base',a.baseline(ids)),('trial',p)]:
  s=a.summarize(data)
  print(Path(path).stem,name, 'errors',s['errors'])
  for mode in ['auto','full']:
   t=s[mode]
   print(mode,'maps/accept',t['fit'],t['accepted'],[(v,round(t[v]['error']['median'],3),round(t[v]['error']['p90'],3),(t[v]['win'],t[v]['lose'],t[v]['tie'])) for v in ['candidate','returned']])
