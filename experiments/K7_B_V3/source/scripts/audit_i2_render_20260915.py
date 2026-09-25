"""Read-only Protocol-A audit; outputs a new report, never changes training PNGs.

Run with the isolated Pillow/FreeType environment used by prepare_i_cn.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, features
from scripts.build_cn2west_v2_proto_abc import find_size_A, render_glyph_AB


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/root/projects/hrfont'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = args.root / 'data/fontdiffuser-p253-t295-s338-cn2west-v2'
    extension = args.root / 'artifacts/i_20260915/cn_content'
    protocol = json.loads((data / 'summary.json').read_text())['protocol']
    manifest = json.loads((extension / 'COMPLETE.json').read_text())
    charset = json.loads((args.root / 'manifests/charset_cn2west_v2_planned.json').read_text())
    han = list(charset['style_han_338'])
    legacy = sorted((data / 'train/ContentImage').glob('*.png'))
    west = [chr(int(p.stem[1:], 16)) for p in legacy]
    font = protocol['content_font']
    sizes = {k: find_size_A(font, chars) for k, chars in
             [('target295', west), ('han338_only', han), ('joint633', west + han)]}
    rows = []
    for ch in han:
        cp = f'u{ord(ch):04X}'
        existing = np.array(Image.open(extension / f'{cp}.png').convert('RGB'))
        golden = np.array(render_glyph_AB(font, ch, sizes['joint633']))
        rows.append(dict(cp=cp, equal=bool(np.array_equal(existing, golden)),
                         hash_matches_manifest=sha(extension / f'{cp}.png') == manifest['neutral_png_sha256'][cp]))
    old = []
    for ch, path in zip(west, legacy):
        a = np.array(Image.open(path).convert('L')).astype(int)
        b = np.array(render_glyph_AB(font, ch, sizes['target295']).convert('L')).astype(int)
        old.append(dict(cp=path.stem, equal=bool(np.array_equal(a, b)), mae=float(np.abs(a-b).mean())))
    result = dict(sizes=sizes, original_content_size=protocol['content_size'],
                  font_sha256=sha(font), pinned_font_matches=sha(font) == manifest['font_sha256'],
                  pillow=Image.__version__, freetype=features.version('freetype2'),
                  cn_n=len(rows), cn_all_equal=all(r['equal'] for r in rows),
                  cn_all_hashes_match=all(r['hash_matches_manifest'] for r in rows), cn_items=rows,
                  legacy_n=len(old), legacy_equal=sum(r['equal'] for r in old),
                  legacy_worst=max(old, key=lambda r: r['mae']), legacy_items=old,
                  interpretation='Han PNGs match joint-inventory Protocol A under the pinned current renderer. Historical western PNG reproduction is a separate check; no legacy images modified.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_items')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
