"""Single source of clean train PNG records for I3 difficulty and E12-c."""
import csv
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = Path('/root/projects/hrfont')
DATA = ROOT/'data/fontdiffuser-p253-t295-s338-cn2west-v2'
CLEAN = ROOT/'manifests/v0913_clean'
PREP = ROOT/'artifacts/i34_20260915'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')


def inventory():
    rows = list(csv.DictReader((CLEAN/'pairs_train.tsv').open(), delimiter='\t'))
    fonts = sorted({r['font'] for r in rows})
    denied = set()
    for sp in ('val', 'test'):
        denied.update(r['font'] for r in csv.DictReader((CLEAN/f'pairs_{sp}.tsv').open(), delimiter='\t'))
    assert not set(fonts) & denied
    records = [dict(font=r['font'], cp=r['cp'], script=r['script_group'], role='target',
                    path=str(DATA/'train/TargetImage'/r['font']/f"{r['font']}+{r['cp']}.png")) for r in rows]
    for font in fonts:
        for p in sorted((DATA/'train/StyleImage'/font).glob('*.png')):
            cp = p.stem[len(font)+1:]
            records.append(dict(font=font, cp=cp, script='han', role='style', path=str(p)))
    assert len({(r['font'], r['role'], r['cp']) for r in records}) == len(records)
    def verify(r):
        p = Path(r['path'])
        with Image.open(p) as im:
            assert im.size == (96, 96) and im.mode == 'RGB', (p, im.size, im.mode)
            im.load()
        return dict(r, sha256=sha(p))
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(verify, records))
    assert all(sum(r['font']==f and r['role']=='style' for r in records) >= 8 for f in fonts)
    return dict(version='clean-train-native96-v1', fonts=fonts, excluded_main_eval_fonts=sorted(denied),
                sources_sha256={name:sha(CLEAN/name) for name in ('INDEX.json','pairs_train.tsv','pairs_val.tsv','pairs_test.tsv','fonts.tsv')},
                records=records)


if __name__ == '__main__':
    dest = PREP/'inventory.json'
    if dest.exists():
        raise SystemExit('Inventory exists; verify/reuse explicitly, do not overwrite')
    obj=inventory(); write_new(dest,obj)
    print(json.dumps(dict(fonts=len(obj['fonts']), records=len(obj['records']), sha256=sha(dest))), flush=True)
