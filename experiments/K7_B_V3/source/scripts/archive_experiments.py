"""Export portable experiment status/inference artifacts; never copy weights or delete runs.

Run on execution server, then rsync reports/experiments/ to the Git staging worktree.
Git publication remains a separate authenticated action; a local export is not a push.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return None

def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/root/projects/hrfont'));a=p.parse_args()
    out=a.root/'reports/experiments'; out.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc).isoformat(); runs=[]
    for run in sorted((a.root/'runs').iterdir()):
        if not run.is_dir() or run.name.startswith('_'):continue
        series=run.name[0] if run.name[0] in 'FGHI' else 'legacy'
        done=read(run/'DONE.json');stopped=read(run/'STOPPED.json');heartbeat=read(run/'heartbeat.json')
        record=dict(run_id=run.name,series=series,server_path=str(run),snapshot_utc=now,
            status='completed' if done else 'stopped' if stopped else 'recorded_not_live_verified',
            done=done,stopped=stopped,heartbeat=heartbeat,config=read(run/'config.json'),inference=[])
        # Archive new I outputs under series/model/step; H portable outputs have a stable legacy path.
        for ev in sorted(run.glob('eval_step_*')):
            marker=read(ev/'DONE.json')
            if not marker:continue
            step=int(ev.name.split('_')[-1]); model=run.name.split('-')[0]
            if series=='I' and not run.name.startswith('I-SMOKE'):
                dest=out/series/model/f'step{step:08d}'
                if not (dest/'ARCHIVE.json').exists():
                    shutil.copytree(ev,dest,dirs_exist_ok=True)
                    files={str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file()}
                    checkpoint=run/f'global_step_{step}/ema.pth'
                    write(dest/'ARCHIVE.json',dict(run_id=run.name,step=step,source=str(ev),files_sha256=files,
                        ema_checkpoint=str(checkpoint),ema_sha256=digest(checkpoint) if checkpoint.is_file() else None))
                path=str(dest.relative_to(a.root/'reports'))+'/review.html'
            elif series=='H':
                path=f'h_20260915/{model}_step{step}/review.html'
            else:
                path=None
            record['inference'].append(dict(step=step,complete=marker,server_path=str(ev),portable_report=path))
        meta=out/series/run.name/'status.json';write(meta,record)
        if (run/'train_log.jsonl').exists():shutil.copy2(run/'train_log.jsonl',meta.parent/'train_log.jsonl')
        runs.append(record)
    write(out/'RUNS.json',dict(snapshot_utc=now,runs=runs,
        note='Heartbeat is an observation, not liveness proof. Official running/queued state is in I queue status; historical absence of DONE is not proof of failure.'))
    q=read(a.root/'reports/i_20260915/status.json')
    if q:write(out/'I/QUEUE.json',q)
    print('EXPERIMENT_EXPORT',len(runs),now)

if __name__=='__main__':main()
