"""Reclaim superseded rolling states after an H arm AND its inference complete.

Inference milestones and the final full optimizer/EMA checkpoint stay intact.
Default is read-only; --apply uses only fixed H run names and exact known files.
"""
import argparse
import json
from pathlib import Path

ROOT=Path('/root/projects/hrfont')
ARMS=['H3','H0','H4','H2','H1','H-D+','H-D-']
FILES={'model.pth','ema.pth','trainer.pt','checkpoint.json'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    records=[]
    for arm in ARMS:
        out=ROOT/'runs'/f'{arm}-V0915-S3407'
        if not (out/'DONE.json').exists():
            continue
        done=json.loads((out/'DONE.json').read_text())
        if done.get('step')!=10000 or not done.get('inference_complete'):
            continue
        final=(out/'last_state').resolve(strict=True)
        assert final==out/'state_step_10000'
        for old in out.glob('state_step_*'):
            if old==final:
                continue
            assert old.parent==out and not old.is_symlink() and old.is_dir()
            assert {p.name for p in old.iterdir()}==FILES
            assert json.loads((old/'checkpoint.json').read_text())['step']<10000
            # Skip any checkpoint explicitly referenced by an active command.
            active=False
            for proc in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    active |= str(old).encode() in proc.read_bytes()
                except OSError:
                    pass
            if active:
                continue
            records.append(dict(path=str(old),bytes=sum((old/f).stat().st_size for f in FILES)))
            if args.apply:
                for f in FILES:
                    (old/f).unlink()
                old.rmdir()
    print(json.dumps(dict(applied=args.apply,records=records),indent=2))


if __name__=='__main__':
    main()
