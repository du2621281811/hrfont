"""One-shot, manifest-gated cleanup of completed G runs; never remove run dirs."""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path('/root/projects/hrfont')
RUNS = ROOT / 'runs'
OLD = [
    'G0b-F0-V0913-BS256-A-S3407', 'G0c-F0-V0913-BS256-A-S3407',
    'G1-F1-V0913-A-S3407', 'G2-F2-V0913-A-S3407', 'G2-RL-V0913-A-S3407',
    'G-CONT-G2-8gpu-V0914-A-S3407', 'G-CONT-G2RL-8gpu-V0914-A-S3407',
    'G-TC-G2-8gpu-V0913-A-S3407', 'G-TC-G2RL-8gpu-V0913-A-S3407',
    'G-RL-pilot-V0913-A-S3407', 'G-RL-pilot-8gpu-V0913-A-S3407',
    'G-REF-SMOKE-A1-T2-V0914-S3407', 'G-REF-SMOKE-B1-T2-V0914-S3407',
]
RESUME = {
    'G0b-F0-V0913-BS256-A-S3407/global_step_10000',
    'G0c-F0-V0913-BS256-A-S3407/global_step_20000',
    'G-CONT-G2-8gpu-V0914-A-S3407/global_step_5000',
}
REPORT = ROOT / 'reports/g_storage_prune_20260915.json'


def inventory():
    candidates = []
    for name in OLD:
        run = RUNS / name
        done = json.loads((run / 'DONE.json').read_text())
        assert done['status'] == 'completed', name
        for ckpt in run.iterdir():
            if not ckpt.is_dir() or ckpt.is_symlink():
                continue
            if not (ckpt.name in ('last_state', 'best') or ckpt.name.startswith('global_step_')):
                continue
            for f in ckpt.iterdir():
                if not f.is_file() or f.is_symlink() or f.suffix not in ('.pt', '.pth'):
                    continue
                reason = None
                if ckpt.name == 'last_state':
                    reason = 'completed smoke or redundant final snapshot; milestones retained'
                elif name.startswith('G-RL-pilot') and ckpt.name == 'global_step_5000':
                    reason = 'pilot final not selected; measured best milestone retained'
                elif f.name == 'trainer_state.pt' and str(ckpt.relative_to(RUNS)) not in RESUME:
                    reason = 'inference/warm-start only; historical optimizer no longer needed'
                if reason:
                    st = f.stat()
                    candidates.append(dict(path=str(f), size=st.st_size, mtime_ns=st.st_mtime_ns, reason=reason))
    targets = {Path(x['path']) for x in candidates}
    for base in (RUNS, ROOT / 'artifacts'):
        for directory, dirs, files in os.walk(base, followlinks=False):
            for name in dirs + files:
                p = Path(directory) / name
                if p.is_symlink():
                    resolved = p.resolve()
                    assert not any(resolved == t or resolved in t.parents for t in targets), str(p)
    # Active workers must not mention a deletion target, including checkpoint dirs.
    deleted_dirs = {str(p.parent) for p in targets if p.name != 'trainer_state.pt'}
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        assert not any(str(p) in cmd for p in targets), cmd
        assert not any(p in cmd for p in deleted_dirs), cmd
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    files = inventory()
    if args.apply:
        original = json.loads(REPORT.read_text())
        assert original['state'] == 'DRY_RUN'
        assert original['files'] == files, 'Inventory changed; re-review dry run'
        before = shutil.disk_usage(ROOT).free
        for item in files:
            p = Path(item['path'])
            assert p.is_relative_to(RUNS) and p.parent.parent.name in OLD
            assert not p.is_symlink()
            st = p.stat()
            assert (st.st_size, st.st_mtime_ns) == (item['size'], item['mtime_ns'])
            p.unlink()
        original.update(state='APPLIED', free_before=before, free_after=shutil.disk_usage(ROOT).free)
        REPORT.write_text(json.dumps(original, indent=2) + '\n')
        print(json.dumps({k:v for k,v in original.items() if k != 'files'}, indent=2))
    else:
        assert not REPORT.exists(), 'Existing audit report must not be overwritten'
        payload = dict(state='DRY_RUN', files=files, logical_bytes=sum(x['size'] for x in files),
                       retained_resume=sorted(RESUME), count=len(files))
        REPORT.write_text(json.dumps(payload, indent=2) + '\n')
        for name in OLD:
            selected = [x for x in files if Path(x['path']).parent.parent.name == name]
            print(name, len(selected), round(sum(x['size'] for x in selected)/2**30, 3), 'GiB')
        print('TOTAL', len(files), payload['logical_bytes']/2**30, 'GiB')


if __name__ == '__main__':
    main()
