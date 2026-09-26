"""Frozen E12c scoring of existing I-series images; never invokes a generator.

Prepare on the verified publication archive. Score on the execution host using
the exact frozen E12c modules. Fixed evaluator REF8, font-macro reporting;
GT is scored separately as a diagnostic and never enters the generated query head.
"""
import argparse
import collections
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path('/root/projects/hrfont')
DATA = ROOT / 'data/fontdiffuser-p253-t295-s338-cn2west-v2'
REF8 = [f'u{ord(c):04X}' for c in '永和书风骨韵天地']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        return json.load(f)


def write(path, obj):
    path = Path(path)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def codepoint(value):
    return value if value.startswith('u') and len(value) >= 5 else f'u{ord(value):04X}'


def prepare(args):
    cp_group = read(args.archive_root / 'preflight_i56/detail_manifest.json')['cp_group']
    jobs, groups = [], []
    for arm in ('I0', 'I1', 'I2', 'I3', 'I4', 'I5'):
        folders = sorted((args.archive_root / arm).glob('step*'))
        assert folders, ('Missing model archive', arm)
        final_weight = None
        final = args.archive_root / arm / 'step00010000_k1248/protocol.json'
        if final.exists():
            final_weight = read(final)['weight_sha256']
        for folder in folders:
            archive = read(folder / 'ARCHIVE.json')
            protocol = read(folder / 'protocol.json')
            rows = read(folder / 'metrics.json')['rows']
            step = int(folder.name[4:].split('_')[0])
            kind = 'matched4888' if folder.name.endswith('_k1248') else ('formal4096' if step == 10000 else 'val192')
            expected = {'matched4888': 4888, 'formal4096': 4096, 'val192': 192}[kind]
            assert len(rows) == expected
            weight = protocol.get('weight_sha256') or archive.get('ema_sha256') or final_weight
            assert weight and len(weight) == 64, (arm, folder, 'weight provenance')
            source = ROOT / 'reports/experiments/I' / arm / folder.name
            if arm == 'I5' or (arm == 'I2' and kind != 'matched4888'):
                run = 'I5-V0916-S3407' if arm == 'I5' else 'I2-V0915-S3407'
                source = ROOT / 'runs' / run / f'eval_step_{step}'
            group_id = f'{arm}/{folder.name}'
            groups.append(dict(group_id=group_id, model=arm, step=step, protocol=kind,
                               count=len(rows), source=str(source), generator_weight_sha256=weight,
                               archive_sha256=sha(folder / 'ARCHIVE.json'),
                               generation_protocol_sha256=sha(folder / 'protocol.json')))
            for r in rows:
                split = r.get('split') or protocol['split']
                assert split in ('train', 'val', 'test')
                gt = r.get('gt') or f"GT__{r['font']}__{r['cp']}.png"
                assert r['cp'] in cp_group
                for name in (r['png'], gt):
                    assert not Path(name).is_absolute() and '..' not in Path(name).parts
                    assert name in archive['files_sha256'] and (folder / name).is_file()
                jobs.append(dict(sample_id=f"{group_id}/{r['png']}", group_id=group_id,
                    model=arm, step=step, protocol=kind, split=split, font=r['font'], cp=r['cp'],
                    script=cp_group[r['cp']], generation_shot=r['k'], generation_group=r.get('group', 0),
                    generation_refs=[codepoint(v) for v in r['refs']], generation_seed=r['seed'],
                    query_path=str(source / r['png']), query_sha256=archive['files_sha256'][r['png']],
                    gt_path=str(source / gt), gt_sha256=archive['files_sha256'][gt]))
    assert len({r['sample_id'] for r in jobs}) == len(jobs)
    manifest = dict(version='e12c-existing-i-fixedref8-v1', eval_refs=REF8, groups=groups, jobs=jobs,
        aggregation='Within each protocol/split/step/shot/script: mean within font, then equal-font mean. ALL averages characters within font, not equal-script averaging.',
        missing='Fail on missing/corrupt/mismatched image; I5 only2k/4k, I6 absent; never synthesize missing outputs.',
        primary_score='Frozen E12c full-head logit, higher compatibility, not calibrated probability',
        gt_policy='Separate real-GT scoring for context; never an input to generated-image score; not an upper bound')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, 'xt', encoding='utf8') as f:
        json.dump(manifest, f, ensure_ascii=False, separators=(',', ':'))
    print(json.dumps(dict(groups=len(groups), images=len(jobs), manifest_sha256=sha(args.out))))


def gpu_admission():
    baseline = {1397110, 1397113, 1397116, 1397119, 1397122, 1397125, 1397128, 1397131}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    pids = {int(s) for s in raw.splitlines() if s.strip()}
    assert pids <= baseline, ('Other GPU work active; no termination', sorted(pids - baseline))
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [[int(s.strip()) for s in line.split(',')] for line in raw.splitlines()]
    assert rows[0][1] >= 8000 and rows[0][2] <= 5
    return dict(preserved_pids=sorted(pids), gpu0=rows[0])


def summarize(rows, out):
    fields = ['model', 'step', 'protocol', 'split', 'generation_shot', 'script']
    cells = collections.defaultdict(list)
    for r in rows:
        for shot in (str(r['generation_shot']), 'ALL'):
            for script in (r['script'], 'ALL'):
                cells[(r['model'], r['step'], r['protocol'], r['split'], shot, script)].append(r)
    summaries, per_font = [], []
    for key, records in sorted(cells.items()):
        fonts = collections.defaultdict(list)
        for r in records:
            fonts[r['font']].append(r)
        details = []
        for font, items in sorted(fonts.items()):
            d = dict(zip(fields, key), font=font, n=len(items),
                logit=statistics.mean(r['logit'] for r in items),
                cosine=statistics.mean(r['cosine_diagnostic'] for r in items),
                gt_logit=statistics.mean(r['gt_logit_diagnostic'] for r in items))
            per_font.append(d)
            details.append(d)
        scores = [d['logit'] for d in details]
        summaries.append(dict(zip(fields, key), n=len(records), fonts=len(fonts),
            logit=statistics.mean(scores), font_sd=statistics.stdev(scores) if len(scores) > 1 else 0.,
            cosine=statistics.mean(d['cosine'] for d in details),
            gt_logit=statistics.mean(d['gt_logit'] for d in details)))
    for name, records in (('summary.csv', summaries), ('per_font.csv', per_font)):
        with (out / name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(out / 'summary.json', summaries)


def score(args):
    sys.path.insert(0, str(args.frozen_code / 'scripts'))
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from e12c_model import Encoder, SetHead

    class Pixels(Dataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            with Image.open(self.paths[i]) as im:
                assert im.size == (96, 96), self.paths[i]
                x = torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2, 0, 1).float() / 255
            return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]

    assert not args.out.exists(), 'Immutable evaluation output already exists'
    with (ROOT / 'reports/i_20260915/queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission = gpu_admission()
        assert shutil.disk_usage(ROOT).free > 10 * 2**30
        start = time.time();m = read(args.manifest);done = read(args.checkpoint / 'DONE.json')
        config = read(args.checkpoint / 'config.json');train = read(args.training_manifest)
        assert done['status'] == 'completed' and not done['smoke']
        assert sha(args.training_manifest) == done['manifest_sha256']
        assert sha(args.checkpoint / 'A_best.pt') == done['encoder_sha256']
        assert sha(args.checkpoint / 'B_best.pt') == done['head_sha256']
        for name, digest in config['code_sha256'].items():
            assert sha(args.frozen_code / 'scripts' / name) == digest, name
        assert m['eval_refs'] == REF8
        paths, expected, refs = set(), {}, {}
        for r in m['jobs']:
            if r['split'] in ('test', 'val'):
                assert r['font'] not in train['split'], ('Evaluator font leakage', r['font'])
            for pathkey, hashkey in (('query_path', 'query_sha256'), ('gt_path', 'gt_sha256')):
                p = r[pathkey];assert p.startswith(str(ROOT) + '/')
                assert p not in expected or expected[p] == r[hashkey]
                expected[p] = r[hashkey];paths.add(p)
            key = (r['split'], r['font'])
            refs[key] = [str(DATA / r['split'] / 'StyleImage' / r['font'] / f"{r['font']}+{cp}.png") for cp in REF8]
            paths.update(refs[key])
        # Verify every scored byte against the publication archive before GPU execution.
        for p, digest in expected.items():
            assert sha(p) == digest, ('Archive image mismatch', p)
        ref_hash = {p:sha(p) for values in refs.values() for p in values}
        torch.set_num_threads(4);torch.manual_seed(3407)
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda:0');e = Encoder(False).to(device).eval();h = SetHead().to(device).eval()
        state_a = torch.load(args.checkpoint / 'A_best.pt', map_location=device, weights_only=False)
        state_b = torch.load(args.checkpoint / 'B_best.pt', map_location=device, weights_only=False)
        assert state_b['encoder_sha256'] == done['encoder_sha256']
        e.load_state_dict(state_a['model']);h.load_state_dict(state_b['head'])
        args.out.mkdir(parents=True)
        write(args.out / 'protocol.json', dict(version=m['version'], manifest_sha256=sha(args.manifest),
            encoder_sha256=done['encoder_sha256'], head_sha256=done['head_sha256'],
            training_manifest_sha256=done['manifest_sha256'], evaluation_script_sha256=sha(__file__),
            eval_refs=REF8, preprocessing=config['preprocessing'], precision='FP32, matching score_e12c.py; no AMP',
            evaluator_split_counts=train['split_counts'], known_family_fonts=len(train['known_families']),
            aggregation=m['aggregation'], primary_score=m['primary_score'], gt_policy=m['gt_policy'],
            gpu_admission=admission, reference_png_sha256=ref_hash,
            selected_A_step=state_a['step'], selected_B_step=state_b['step'],
            train_outputs='Diagnostic only; main generator train fonts can overlap E12c training pool',
            human_agreement='Not yet validated; no probability threshold or quality-pass classification'))
        ordered = sorted(paths);lookup = {p:i for i,p in enumerate(ordered)}
        features = torch.empty((len(ordered), 512), dtype=torch.float32)
        loader = DataLoader(Pixels(ordered), batch_size=128, num_workers=4, pin_memory=True)
        cursor = 0
        with torch.inference_mode():
            for batch in loader:
                z = e(batch.to(device, non_blocking=True)).cpu();features[cursor:cursor+len(z)] = z;cursor += len(z)
                if cursor % 4096 == 0:print('ENCODE', cursor, len(ordered), flush=True)
            def evaluate(jobs, target='query_path'):
                q = features[[lookup[r[target]] for r in jobs]].to(device)
                rr = torch.stack([features[[lookup[p] for p in refs[r['split'], r['font']]]] for r in jobs]).to(device)
                return h(q, rr, torch.ones(len(jobs), 8, dtype=torch.bool, device=device))
            first = m['jobs'][:8];batched = evaluate(first)[0]
            single = torch.cat([evaluate([r])[0] for r in first])
            assert torch.allclose(batched, single, atol=1e-4, rtol=1e-5), 'Batch/head mismatch'
            # Direct uncached FP32 single-query path must match the existing CLI convention.
            sample = first[0];direct_paths = [sample['query_path']] + refs[sample['split'],sample['font']]
            pixels = Pixels(direct_paths);z = e(torch.stack([pixels[i] for i in range(9)]).to(device))
            direct = h(z[:1], z[1:][None], torch.ones(1,8,dtype=torch.bool,device=device))[0]
            assert torch.allclose(direct, batched[:1], atol=1e-4, rtol=1e-5), 'Direct/cache mismatch'
            perm = h(z[:1], z[1:].flip(0)[None], torch.ones(1,8,dtype=torch.bool,device=device))[0]
            assert torch.allclose(direct, perm, atol=1e-5, rtol=1e-5)
            one = h(z[:1],z[1:2][None],torch.ones(1,1,dtype=torch.bool,device=device))[0]
            dup = h(z[:1],z[1:2].repeat(8,1)[None],torch.ones(1,8,dtype=torch.bool,device=device))[0]
            assert torch.allclose(one,dup,atol=1e-5,rtol=1e-5)
            results = []
            for start_index in range(0,len(m['jobs']),256):
                jobs = m['jobs'][start_index:start_index+256]
                scores,cos = evaluate(jobs);gt,_ = evaluate(jobs,'gt_path')
                for r,s,c,g in zip(jobs,scores.cpu().tolist(),cos.cpu().tolist(),gt.cpu().tolist()):
                    assert all(math.isfinite(v) for v in (s,c,g))
                    results.append(dict(**r,eval_refs=REF8,logit=s,cosine_diagnostic=c,gt_logit_diagnostic=g,
                        evaluator_font_split=train['split'].get(r['font'],'unseen_main_eval'),lineage=train['known_families'].get(r['font'],'unknown')))
        with gzip.open(args.out/'scores.jsonl.gz','xt',encoding='utf8') as f:
            for r in results:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        summarize(results,args.out)
        write(args.out/'DONE.json',dict(status='completed',images=len(results),groups=len(m['groups']),
            unique_encoded_pngs=len(ordered),seconds=time.time()-start,missing_images=0,
            checks='All query/GT SHA match, fixedREF8 exists, no main val/test font in evaluator pool, direct/cached/batched/permuted/duplicate checks pass'))
        write(args.out/'ARCHIVE.json',dict(files_sha256={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}))
        print('COMPLETED',len(results),flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('score');s.add_argument('--manifest',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    s.add_argument('--checkpoint',type=Path,default=ROOT/'runs/E12C-V0916-S3407')
    s.add_argument('--frozen-code',type=Path,default=Path('/root/projects/hrfont_i34_20260915.bJikwQ'))
    s.add_argument('--training-manifest',type=Path,default=ROOT/'artifacts/i34_20260915/e12c_manifest.json')
    args=ap.parse_args();prepare(args) if args.command=='prepare' else score(args)


if __name__ == '__main__':main()
