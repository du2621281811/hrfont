"""Verify this PI-approved rebuild and optionally activate it before I2 starts.

Preserves the previous content directory and the specific render-review STOP.
Never changes the running I1 code, main PNG tree, or main Ec/Es caches.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--activate', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = Path('/root/projects/hrfont')
    base = root / 'artifacts/i_20260915'
    old = base / 'cn_content'
    fresh = base / 'cn_content_approved_20260915'
    backup = base / 'cn_content_before_approval_20260915'
    out = root / 'reports/i_20260915'
    marker = out / 'STOP'
    marker_backup = out / 'STOP_RENDER_REVIEW_APPROVED_20260915.txt'
    m0, m1 = [json.loads((p / 'COMPLETE.json').read_text()) for p in (old, fresh)]
    for key in ('n', 'chars', 'font_sha256', 'parent_ec_sha256', 'content_size', 'pillow', 'freetype'):
        assert m0[key] == m1[key], key
    assert m1['n'] == 338 and m1['content_size'] == 83
    assert sha(Path(m1['source_backend']['content_font'])) == m1['font_sha256']
    for folder, manifest in ((old, m0), (fresh, m1)):
        assert sha(folder / 'features.pt') == manifest['features_sha256']
        assert len(list(folder.glob('u*.png'))) == 338
        for cp in manifest['chars']:
            path = folder / f'{cp}.png'
            assert sha(path) == manifest['neutral_png_sha256'][cp]
            with Image.open(path) as im:
                assert im.mode == 'RGB' and im.size == (96, 96)
    assert m0['neutral_png_sha256'] == m1['neutral_png_sha256']
    f0, f1 = [torch.load(p / 'features.pt', map_location='cpu', weights_only=True) for p in (old, fresh)]
    assert set(f0) == set(f1) == set(m1['chars'])
    count = 0
    for cp in f1:
        assert len(f0[cp]) == len(f1[cp]) == 5
        for a, b in zip(f0[cp], f1[cp]):
            assert a.dtype == b.dtype == torch.float16 and torch.isfinite(b).all()
            assert torch.equal(a, b), cp
            count += 1
    result = dict(status='VERIFIED', time_utc=datetime.now(timezone.utc).isoformat(),
                  authorization='PI message 2026-09-15 approves Protocol A Han rebuild and I2 scheduling',
                  png_exact=338, feature_tensors_exact=count,
                  font_sha256=m1['font_sha256'], parent_ec_sha256=m1['parent_ec_sha256'],
                  features_sha256=m1['features_sha256'],
                  scope='New Han neutral content only; old western PNG and main caches unchanged',
                  historical_western_pixel_reproduction='unresolved; accepted as separate from the approved Han extension',
                  activated=False)
    if args.activate:
        assert not backup.exists() and not marker_backup.exists()
        assert not (root / 'runs/I2-V0915-S3407').exists(), 'I2 already started; do not swap data'
        state = json.loads((out / 'status.json').read_text())
        assert state['state'] == 'RUNNING' and state['task'] == 'I1', state
        expected = ('2026-09-15: User requires I2 Han rendering to match the original Git construction method and review the construction. '
                    'Let I1 finish normally. Hold subsequent formal I2 training until the renderer audit and user review are resolved. '
                    'Queue STOP is not a request to stop I1.')
        assert marker.read_text().strip() == expected, 'STOP changed; manual review required'
        old.rename(backup)
        try:
            fresh.rename(old)
        except Exception:
            backup.rename(old)
            raise
        # Release only after successful asset activation. Any error leaves STOP in place.
        result.update(activated=True, active_path=str(old), backup_path=str(backup),
                      queue_order=['I1 training and inference', 'I2 preflight then training and inference', 'I0 inference'],
                      stop_archived_as=str(marker_backup))
        (out / 'I2_RENDER_APPROVAL_20260915.json').write_text(json.dumps(result, indent=2) + '\n')
        marker.rename(marker_backup)
    else:
        (out / 'I2_REBUILD_VERIFICATION_20260915.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
