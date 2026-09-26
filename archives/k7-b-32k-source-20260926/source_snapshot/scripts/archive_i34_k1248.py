"""Archive completed I k1248 boards with exact protocol, images and pixel metrics.

CPU-only; never touches model weights. Existing immutable archives are not overwritten.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path('/root/projects/hrfont')
SOURCE = ROOT / 'reports/g_v0913_shot_k1248'
DATA = ROOT / 'data/fontdiffuser-p253-t295-s338-cn2west-v2'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def gray(p):
    with Image.open(p) as im:
        assert im.size == (96, 96)
        return np.asarray(im.convert('L'), dtype='float32') / 255


def edge(im):
    return np.diff(im, axis=0, prepend=im[:1])**2 + np.diff(im, axis=1, prepend=im[:, :1])**2


def archive(arm, run_id):
    protocol = json.loads((SOURCE / 'PROTOCOL.json').read_text())
    dest = ROOT / 'reports/experiments/I' / arm / 'step00010000_k1248'
    if (dest / 'ARCHIVE.json').exists():
        print(f'{arm}: existing archive; verify before reuse', flush=True)
        return
    assert not dest.exists(), f'Partial archive requires review: {dest}'
    assert shutil.disk_usage(ROOT).free > 10 * 2**30
    pairs = [(split, font) for split, key in [('test', 'test'), ('train', 'train5'), ('val', 'val5')]
             for font in protocol['fonts'][key]]
    assert len(pairs) == 26 and len(protocol['chars']) == 47
    for k in (1, 2, 4, 8):
        done = json.loads((SOURCE / 'preds' / f'{arm}_s{k}' / 'DONE.json').read_text())
        assert done['images'] == done['expected'] == 1222 and done['status'] == 'completed'
    dest.mkdir(parents=True)
    for name in ('preds', 'gt', 'refs', 'content', 'pages'):
        (dest / name).mkdir()
    shutil.copy2(SOURCE / 'PROTOCOL.json', dest / 'SOURCE_PROTOCOL.json')
    ckpt = ROOT / 'runs' / run_id / 'global_step_10000'
    weight = ckpt / ('unet.pth' if arm == 'I0' else 'ema.pth')
    local_protocol = {k: protocol[k] for k in ('ref8', 'shot_sets', 'sampling', 'fonts', 'chars')}
    local_protocol.update(arm=arm, step=10000, checkpoint=str(ckpt), weight=weight.name,
                          weight_sha256=sha(weight), source_protocol_sha256=sha(SOURCE / 'PROTOCOL.json'),
                          protocol_id='G-k1248-test16-train5-val5-47chars',
                          metric_definition='grayscale[0,1]; L1; SSIM win7; squared finite-difference edge L1, same formula as i_eval.py',
                          split_policy='test reported separately; train/val are diagnostics; do not pool with formal4096 protocol')
    write(dest / 'protocol.json', local_protocol)
    rows = []
    index = ['<!doctype html><meta charset="utf-8"><h1>' + arm + ' @10k — k1248</h1>',
             '<p>26 fonts ×47 chars ×4 shots. DPM++20 / CFG7.5 / seed3407. Test/train/val separated.</p><ul>']
    for split, font in pairs:
        title = html.escape(f'{split} / {font}')
        page = ['<!doctype html><meta charset="utf-8"><style>body{font:14px sans-serif}img{width:96px}td{padding:4px}th{position:sticky;top:0;background:white}</style>',
                f'<a href="../review.html">Index</a><h1>{html.escape(arm)} {title}</h1><p>Ref8:']
        for ch in protocol['ref8']:
            cp = f'u{ord(ch):04X}'
            src = DATA / split / 'StyleImage' / font / f'{font}+{cp}.png'
            ref = f'{split}__{font}__{cp}.png'
            shutil.copy2(src, dest / 'refs' / ref)
            page.append(f'<img loading="lazy" src="../refs/{html.escape(ref, quote=True)}">')
        page.append('</p><table><tr><th>char</th><th>Content</th><th>GT</th><th>1 shot</th><th>2 shot</th><th>4 shot</th><th>8 shot</th></tr>')
        for ch in protocol['chars']:
            cp = f'u{ord(ch):04X}'
            gt = DATA / split / 'TargetImage' / font / f'{font}+{cp}.png'
            if not gt.exists():
                gt = next(DATA / sp / 'TargetImage' / font / f'{font}+{cp}.png'
                          for sp in ('test', 'train', 'val') if (DATA / sp / 'TargetImage' / font / f'{font}+{cp}.png').exists())
            gt_name = f'{split}__{font}__{cp}.png'
            shutil.copy2(gt, dest / 'gt' / gt_name)
            content = next(DATA / sp / 'ContentImage' / f'{cp}.png'
                           for sp in ('test', 'val', 'train') if (DATA / sp / 'ContentImage' / f'{cp}.png').exists())
            if not (dest / 'content' / f'{cp}.png').exists():
                shutil.copy2(content, dest / 'content' / f'{cp}.png')
            g = gray(gt)
            cells = [f'<tr><td>{html.escape(ch)} {cp}</td><td><img loading="lazy" src="../content/{cp}.png"></td>',
                     f'<td><img loading="lazy" src="../gt/{html.escape(gt_name, quote=True)}"></td>']
            for k in (1, 2, 4, 8):
                source = SOURCE / 'preds' / f'{arm}_s{k}' / split / font / f'{split}__{font}__{cp}__s3407.png'
                name = f'{split}__{font}__{cp}__k{k}.png'
                shutil.copy2(source, dest / 'preds' / name)
                p = gray(dest / 'preds' / name)
                rows.append(dict(arm=arm, split=split, font=font, cp=cp, k=k, seed=3407,
                                 refs=protocol['ref8'][:k], png='preds/' + name, gt='gt/' + gt_name,
                                 source=str(source), l1=float(np.abs(p-g).mean()),
                                 ssim=float(structural_similarity(p,g,data_range=1.,win_size=7)),
                                 edge_l1=float(np.abs(edge(p)-edge(g)).mean())))
                cells.append(f'<td><img loading="lazy" src="../preds/{html.escape(name, quote=True)}"></td>')
            page.append(''.join(cells) + '</tr>')
        page.append('</table>')
        page_name = f'{split}__{font}.html'
        (dest / 'pages' / page_name).write_text('\n'.join(page))
        index.append(f'<li><a href="pages/{html.escape(page_name, quote=True)}">{title}</a></li>')
    assert len(rows) == 4888 and len({(r['split'],r['font'],r['cp'],r['k']) for r in rows}) == 4888
    summary = {sp: {str(k): {m: float(np.mean([r[m] for r in rows if r['split']==sp and r['k']==k]))
                             for m in ('l1','ssim','edge_l1')} for k in (1,2,4,8)} for sp in ('test','train','val')}
    write(dest / 'metrics.json', dict(summary=summary, rows=rows))
    (dest / 'review.html').write_text('\n'.join(index) + '</ul>')
    write(dest / 'ARCHIVE.json', dict(arm=arm, images=4888, time=time.time(), source=str(SOURCE),
          files_sha256={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(arm=arm, archived=4888, root=str(dest), summary=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=['I3','I4'], required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    assert len(args.arms)==1
    archive(args.arms[0],args.run_id)
