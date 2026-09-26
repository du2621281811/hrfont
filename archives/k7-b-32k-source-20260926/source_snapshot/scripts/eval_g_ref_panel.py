#!/usr/bin/env python3
"""Isolated clean-protocol P0 / fixed-val panel, eight GPU workers maximum."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
ROOT = Path('/root/projects/hrfont')
PY = '/root/miniforge3/envs/boogu/bin/python'


def worker(job, out):
    # Fail before loading eight GPU models if metric dependencies are missing.
    from PIL import Image
    import numpy as np
    from skimage.metrics import structural_similarity
    sys.path.insert(0, str(CODE / 'scripts'))
    import eval_f03_test16_strat as E
    from v0913_clean_lib import load_phase_pairs
    E.configure_split('val', out=out)
    E.REF8 = list(job['refs'])
    E.REF8_CPS = [E.cp_of(c) for c in E.REF8]
    E.STYLE1_CPS = E.REF8_CPS[:1]
    E.STRATIFIED = job['chars']
    spec = dict(label=job['id'], kind='f2', variant=CODE / 'code/variants/cn2west_f123_rsi/FontDiffuser',
                ckpt=Path(job['checkpoint']), style_k=job['k'], style_rl128=job['rl'],
                style_pattn=False, tc_enabled=job.get('tc', False),
                local_count_norm=job.get('n', False), clean_map=str(ROOT / 'manifests/v0913_clean'),
                es_cache=ROOT / 'artifacts/g0/es_spatial', ec_cache=ROOT / 'artifacts/g0/ec_multiscale',
                es_local_cache=ROOT / 'artifacts/g0/es_local', tc_cache=str(ROOT / 'artifacts/tc_v2_cache'))
    E.METHODS[job['id']] = spec
    complete = all(E.pred_path(job['id'], font, char).is_file()
                   for font in job['fonts'] for char in job['chars'])
    if not complete:
        E.generate_f2('cuda:0', job['fonts'], False, mid=job['id'])
    allowed = {(r['font'], r['char']) for r in load_phase_pairs(ROOT / 'manifests/v0913_clean', 'val')[0]}
    rows = []
    for font in job['fonts']:
        for char in job['chars']:
            if (font, char) not in allowed:
                continue
            p = E.pred_path(job['id'], font, char)
            if not p.is_file():
                raise RuntimeError(f'missing expected prediction: {p}')
            pred = np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255
            gt = np.asarray(Image.open(E.gt_path(font, char)).convert('L'), dtype=np.float32) / 255
            rows.append(dict(font=font, char=char, l1=float(abs(pred-gt).mean()),
                             ssim=float(structural_similarity(pred, gt, data_range=1., win_size=7))))
    (out / (job['id'] + '.metrics.json')).write_text(json.dumps(dict(job=job, rows=rows), indent=2) + '\n')


def main():
    from skimage.metrics import structural_similarity  # preflight, before worker dispatch
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--worker', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--rl', action='store_true')
    ap.add_argument('--tc', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker(json.loads(a.worker.read_text()), a.out)
        return
    fonts = sorted(json.loads((ROOT / 'manifests/split_v3_228_16_16.json').read_text())['stems']['val'])[:4]
    old = '永和书风骨韵天地'
    jobs = []
    if a.checkpoint:
        for k in (1, 2, 4, 8):
            for seq, refs in enumerate((old, old[::-1])):
                jobs.append(dict(id=f'k{k}_refs{seq}', checkpoint=str(a.checkpoint), rl=a.rl,
                                 k=k, refs=refs, tc=a.tc))
    else:
        for rl, parent in ((False, 'G2-F2-V0913-A-S3407'), (True, 'G2-RL-V0913-A-S3407')):
            for case, refs, k, n in [('one', old, 1, False), ('duplicate8', '永'*8, 8, False),
                                     ('distinct8', old, 8, False),
                                     ('normalized_dup8' if rl else 'permuted8', '永'*8 if rl else old[::-1], 8, rl)]:
                jobs.append(dict(id=f'{"RL" if rl else "G2"}_{case}',
                                 checkpoint=str(ROOT / 'runs' / parent / 'global_step_10000'),
                                 rl=rl, refs=refs, k=k, n=n))
    for j in jobs:
        j.update(fonts=fonts, chars=['0', 'A', 'g', 'é'])
    (a.out / 'protocol.json').write_text(json.dumps(dict(jobs=jobs, seed=3407, steps=20, cfg=7.5,
        purpose='small val diagnostic; not the full four-reference-set benchmark'), ensure_ascii=False, indent=2) + '\n')
    children = []
    logs = []
    for gpu, job in enumerate(jobs):
        jp = a.out / (job['id'] + '.job.json')
        if jp.exists() and json.loads(jp.read_text()) != job:
            raise RuntimeError(f'existing panel job has different inputs: {jp}')
        jp.write_text(json.dumps(job, ensure_ascii=False, indent=2) + '\n')
        log = (a.out / (job['id'] + '.log')).open('a')
        logs.append(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1')
        children.append(subprocess.Popen([PY, str(Path(__file__).resolve()), '--out', str(a.out),
                                          '--worker', str(jp)], env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in children]
    for log in logs:
        log.close()
    if any(codes):
        raise RuntimeError(f'panel workers failed: {codes}')
    (a.out / 'DONE.json').write_text(json.dumps(dict(status='completed', jobs=len(jobs))) + '\n')


if __name__ == '__main__':
    main()
