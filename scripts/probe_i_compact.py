"""Read-only bounded I-series heartbeat probe; no training control or imports."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/root/projects/hrfont')
OUT = ROOT / 'reports/i_20260915'
RUN = ROOT / 'runs/I2-V0915-S3407'


def small_json(path):
    try:
        if path.stat().st_size > 16384:
            return {'probe_error': 'oversized_json'}
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as exc:
        return {'probe_error': type(exc).__name__}


def tail(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            lines = f.read().decode('utf-8', errors='replace').splitlines()
        return lines[-12:]
    except FileNotFoundError:
        return []


def latest_rank(path):
    for line in reversed(tail(path)):
        try:
            row = json.loads(line)
            return row
        except ValueError:
            continue
    return None


def select(row, keys):
    return {k: row[k] for k in keys if k in row} if isinstance(row, dict) else row


def probe():
    ranks = [latest_rank(RUN / f'rank{i}.jsonl') for i in range(8)]
    rows = [r for r in ranks if r is not None]
    health = {'ranks_read': len(rows)}
    for field in ('step', 'attempt', 'skips', 'scale', 'ddp_spread', 'reader_grad', 'encoder_grad', 'router_grad'):
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
        if values:
            health[field] = [min(values), max(values)] if all(math.isfinite(v) for v in values) else 'NONFINITE'
    if rows:
        health['lr'] = rows[0].get('lr')
    processes = subprocess.run(['ps', '-eo', 'pid=,comm=,args='], capture_output=True, text=True, check=True)
    # Print only coordinators, not eight duplicate workers or full command lines.
    live = []
    for line in processes.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid, comm, args = parts
        if not (comm.startswith('python') or comm == 'pt_elastic'):
            continue
        if 'torch.distributed.run' in args:
            scripts = [p.rsplit('/', 1)[-1] for p in args.split()
                       if p.endswith('.py') and ('train_i' in p or 'eval_i' in p)]
            if scripts:
                live.append({'pid': int(pid), 'script': scripts[0]})
        elif 'queue_i_' in args:
            live.append({'pid': int(pid), 'script': 'I_queue'})
    state = small_json(OUT / 'k1248_infer_status.json')
    task = state.get('task') if isinstance(state, dict) else None
    progress = tail(OUT / f'k1248_{task}.log') if task in ('I0', 'I1', 'I2') else []
    progress = [s[-180:] for s in progress if s.startswith(('INFER ', 'DONE ', 'ARM '))][-2:]
    done = {}
    for arm in ('I0', 'I1', 'I2'):
        done[arm] = {str(k): select(small_json(ROOT / 'reports/g_v0913_shot_k1248/preds' / f'{arm}_s{k}/DONE.json'),
                                   ('status', 'images', 'expected', 'probe_error')) for k in (1, 2, 4, 8)}
    checkpoint = RUN / 'last_state'
    free = shutil.disk_usage(ROOT).free
    hb = RUN / 'heartbeat.json'
    return dict(time=time.time(), free_gib=round(free / 2**30, 2), low_disk=free < 10 * 2**30,
                coordinators=live, queue=select(small_json(OUT / 'status.json'), ('state', 'task', 'error', 'probe_error')),
                current_inference=state, inference_progress=progress, k1248_done=done,
                i2=select(small_json(hb), ('state', 'step', 'attempt', 'probe_error')),
                i2_heartbeat_age_s=round(time.time() - hb.stat().st_mtime) if hb.exists() else None,
                rank_health=health, last_state=checkpoint.resolve().name if checkpoint.exists() else None,
                checkpoint=small_json(checkpoint / 'checkpoint.json'),
                i2_done=small_json(RUN / 'DONE.json'),
                stops={'queue': (OUT / 'STOP').exists(), 'I2': (RUN / 'STOP').exists()},
                note='DONE claims and old heartbeat are hints; verify images/process exit at transitions, read-only probe')


if __name__ == '__main__':
    print(json.dumps(probe(), ensure_ascii=False, separators=(',', ':')))
