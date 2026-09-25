"""Eight-rank disjoint raw12 cache build and train-only teacher calibration."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from h_runtime import ROOT, VARIANT, CACHE, PARENT, T, args_for, dataset, atomic_json, sha256_file
from scripts import hrfont_build_es_local_cache as B


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    args = args_for()
    if rank == 0:
        CACHE.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    if (CACHE / 'manifest.json').exists():
        raise RuntimeError('existing H cache: verify/reuse, never overwrite')
    B.VARIANT = VARIANT
    B.POOL = 12
    split = json.loads(Path(args.split_manifest).read_text())['stems']
    jobs = B.collect_jobs(Path(args.data_root), {k: sorted(split[k]) for k in ('train', 'val', 'test')})
    keys = (Path(args.es_cache_path) / 'keys.txt').read_text().splitlines()
    assert [x[0] for x in jobs] == keys
    es, es_sha = B.load_encoder(PARENT, str(device))
    assert es_sha == json.loads((Path(args.es_cache_path) / 'manifest.json').read_text())['es_checkpoint_sha256']
    n = len(jobs)
    if rank == 0:
        arr = np.memmap(CACHE / 'raw12.dat', mode='w+', dtype=np.float16, shape=(n, 144, 256))
        arr.flush()
        del arr
    dist.barrier()
    arr = np.memmap(CACHE / 'raw12.dat', mode='r+', dtype=np.float16, shape=(n, 144, 256))
    ids = list(range(rank, n, world))
    for start in range(0, len(ids), 32):
        pick = ids[start:start+32]
        arr[pick] = B.encode_batch(es, [jobs[i][1] for i in pick], str(device))
        if start % 1024 == 0:
            print('CACHE', rank, start, len(ids), flush=True)
    arr.flush()
    dist.barrier()
    picks = np.random.default_rng(3407 + rank).choice(ids, 8, replace=False)
    max_error = 0.
    for i in picks:
        live = B.encode_batch(es, [jobs[int(i)][1]], str(device))[0]
        max_error = max(max_error, float(np.abs(arr[i].astype('float32') - live.astype('float32')).max()))
        assert np.allclose(arr[i], live, atol=.002, rtol=.002), i
    print('VERIFY', rank, max_error, flush=True)
    del es
    torch.cuda.empty_cache()
    train = dataset(args, 'train')
    picks = np.random.default_rng(3407).choice(len(train), 2048, replace=False).tolist()
    vgg = T.ContentPerceptualLoss().VGG.to(device).eval().requires_grad_(False)
    total = torch.zeros(257, device=device, dtype=torch.float64)
    mine = picks[rank::world]
    with torch.no_grad():
        for start in range(0, len(mine), 16):
            paths = [train.target_images[i] for i in mine[start:start+16]]
            x = torch.stack([torch.from_numpy(np.array(Image.open(p).convert('RGB'), copy=True)).permute(2,0,1).float()/255 for p in paths]).to(device)
            f = torch.nn.functional.adaptive_avg_pool2d(vgg(T.normalize_mean_std(x))[1], 12).double()
            total[:128] += f.sum((0,2,3))
            total[128:256] += f.square().sum((0,2,3))
            total[-1] += f.shape[0]*144
    dist.all_reduce(total)
    if rank == 0:
        mean = total[:128]/total[-1]
        std = (total[128:256]/total[-1] - mean.square()).clamp_min(1e-4).sqrt()
        torch.save(dict(mean=mean.float().cpu(), std=std.float().cpu()), CACHE / 'teacher_stats.pt')
        torch_hub = Path(torch.hub.get_dir()) / 'checkpoints'
        weights = sorted(torch_hub.glob('vgg16-*.pth'))
        assert len(weights) == 1, weights
        atomic_json(CACHE / 'manifest.json', dict(state='COMPLETE', entries=n, shape=[n,144,256],
            es_sha256=es_sha, keys_sha256=sha256_file(Path(args.es_cache_path)/'keys.txt'),
            payload_sha256=sha256_file(CACHE/'raw12.dat'), teacher_stats_sha256=sha256_file(CACHE/'teacher_stats.pt'),
            teacher_vgg_sha256=sha256_file(weights[0]), teacher='VGG enc_2 spatial pool12 train-only 2048 target calibration',
            teacher_train_paths=[train.target_images[i] for i in picks],
            split_sha256=sha256_file(Path(args.split_manifest)), clean_sha256=T._clean_map_sha256(args.v0913_clean_map),
            created_unix=time.time(), verified_online=8*world))
        print('H_CACHE_COMPLETE', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
