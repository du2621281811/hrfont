"""Validate and seal a copied, completed I-series milestone (CPU only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--run-id', required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--ema-sha256', required=True)
    p.add_argument('--expected', type=int, choices=(192, 4096), default=192)
    a = p.parse_args()
    d = a.directory
    assert not (d / 'ARCHIVE.json').exists(), 'Archive already sealed'
    done = json.loads((d / 'DONE.json').read_text())
    protocol = json.loads((d / 'protocol.json').read_text())
    rows = json.loads((d / 'metrics.json').read_text())['rows']
    assert done['status'] == 'completed' and done['step'] == protocol['step'] == a.step
    assert len(rows) == done['images'] == a.expected
    assert len({r['png'] for r in rows}) == a.expected
    expected_pngs = {r['png'] for r in rows}
    expected_pngs.update(f"GT__{r['font']}__{r['cp']}.png" for r in rows)
    for r in rows:
        assert (d / r['png']).is_file()
        assert all(math.isfinite(r[k]) for k in ('l1', 'ssim', 'edge_l1'))
        assert (d / f"GT__{r['font']}__{r['cp']}.png").is_file()
    images = list(d.glob('*.png'))
    assert {f.name for f in images} == expected_pngs
    for path in images:
        with Image.open(path) as im:
            im.load()
            assert im.size == (96, 96)
    assert (d / 'review.html').is_file()
    files = {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(d.rglob('*')) if f.is_file()}
    record = dict(run_id=a.run_id, step=a.step,
                  source=f'/root/projects/hrfont/runs/{a.run_id}/eval_step_{a.step}',
                  ema_sha256=a.ema_sha256, files_sha256=files,
                  verification=f'{a.expected} predictions and {len(images)-a.expected} GT images decoded at 96x96; finite metrics; local archive hashes')
    with (d / 'ARCHIVE.json').open('x') as out:
        json.dump(record, out, indent=2)
        out.write('\n')
    print(f'{d}: sealed {len(files)} files, {len(images)} decoded PNGs')

if __name__ == '__main__':
    main()
