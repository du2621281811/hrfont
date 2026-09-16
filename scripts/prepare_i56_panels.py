"""CPU-only expansion of existing training review panels to8 CN+8 target examples.

No automatic detail-label approval. Unreviewed groups retain base exposure.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/root/projects/hrfont')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    invpath = ROOT/'artifacts/i34_20260915/inventory.json'
    inv = json.loads(invpath.read_text())
    old = json.loads((ROOT/'artifacts/i34_20260915/difficulty_manifest.json').read_text())
    by = collections.defaultdict(list)
    for r in inv['records']: by[(r['font'], r['script'])].append(r)
    rows = []
    for row in old['review_rows']:
        font, script = row['font'], row['group']
        cn = sorted(by[(font, 'han')], key=lambda r:r['cp'])
        target = sorted(by[(font, script)], key=lambda r:r['cp'])
        chosen_cn = [cn[int(i)] for i in np.linspace(0,len(cn)-1,8)]
        chosen_target = [target[int(i)] for i in np.linspace(0,len(target)-1,8)]
        rows.append(dict(key=font+'|'+script,font=font,script=script,cn=chosen_cn,target=chosen_target))
    for page in range((len(rows)+5)//6):
        canvas = Image.new('RGB',(840,6*234),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(rows[page*6:(page+1)*6]):
            y=j*234;draw.text((2,y),f'{page*6+j}: {row["key"]}',fill='black')
            for line,key in enumerate(('cn','target')):
                for col,r in enumerate(row[key]):
                    path=Path(r['path']);assert sha(path)==r['sha256']
                    with Image.open(path) as im: canvas.paste(im.convert('RGB'),(col*104,y+14+line*109))
                    draw.text((col*104,y+110+line*109),r['cp'],fill='black')
        canvas.save(a.out/f'panel_{page}.png')
    meta=dict(status='REVIEW_PENDING',selection='Existing36 spread-across-Es-score training panels expanded to8+8; not a complete style census',
        inventory_sha256=sha(invpath),sources_sha256=inv['sources_sha256'],train_fonts=inv['fonts'],
        cp_group=old['cp_group'],rows=rows,panel_sha256={p.name:sha(p) for p in sorted(a.out.glob('panel_*.png'))})
    (a.out/'panels.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(groups=len(rows),files=list(meta['panel_sha256']))))


if __name__=='__main__':main()
