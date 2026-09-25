"""Arrange existing H predictions unchanged into compact GT/k1/k4/k8 sheets."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

ap=argparse.ArgumentParser()
ap.add_argument('directory',type=Path)
a=ap.parse_args()
rows=json.loads((a.directory/'metrics.json').read_text())['rows']
fonts=sorted({r['font'] for r in rows})
lookup={(r['font'],r['cp'],r['k']):r for r in rows if r['group']==0}
for page,start in enumerate(range(0,len(fonts),8),1):
    cases=[(f,cp) for f in fonts[start:start+8] for cp in ('u0041','u0067') if (f,cp,1) in lookup]
    canvas=Image.new('RGB',(580,40+len(cases)*108),'#eeeeee')
    d=ImageDraw.Draw(canvas)
    for i,title in enumerate(('GT','k=1','k=4','k=8')):
        d.text((170+i*100,10),title,fill='black')
    for j,(font,cp) in enumerate(cases):
        y=40+j*108
        d.text((3,y+20),font,fill='black')
        d.text((3,y+40),chr(int(cp[1:],16)),fill='black')
        paths=[a.directory/f'GT__{font}__{cp}.png']+[a.directory/lookup[(font,cp,k)]['png'] for k in (1,4,8)]
        for i,p in enumerate(paths):
            canvas.paste(Image.open(p).convert('RGB'),(165+i*100,y))
    dest=a.directory/f'contact_Ag_{page}.png'
    canvas.save(dest)
    print(dest)
