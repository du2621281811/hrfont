"""I2 final EMA adapter for the existing, unchanged I0/I1 k1248 evaluator."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

CODE = Path('/root/projects/hrfont_i_20260915.zvbB1H')
ROOT = Path('/root/projects/hrfont')
sys.path.insert(0, str(CODE / 'scripts'))
sys.path.insert(0, str(CODE))
import eval_i_k1248 as shared


def load_i2(arm, device, scratch):
    assert arm == 'I2'
    run = ROOT / 'runs/I2-V0915-S3407'
    done = json.loads((run / 'DONE.json').read_text())
    assert done['step'] == 10000 and done['inference_complete']
    ckpt = run / 'global_step_10000'
    meta = json.loads((ckpt / 'checkpoint.json').read_text())
    assert meta['complete'] and meta['arm'] == arm and meta['step'] == 10000
    model, args = shared.model_for(arm, device, scratch / 'load')
    model.load_train_state(shared.torch.load(ckpt / 'ema.pth', map_location=device, weights_only=True))
    return model, args, 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', default='1,2,4,8')
    args = ap.parse_args()
    shots = tuple(int(x) for x in args.shots.split(','))
    assert shots == (1, 2, 4, 8)
    assert len(shared.FONT_SPLITS) == 26 and len(shared.CHARS) == 47
    assert ''.join(shared.REF8) == '永和书风骨韵天地'
    assert shared.PROTO['sampling'] == dict(sampler='dpmsolver++', steps=20, cfg=7.5, seed=3407)
    shared.load_model = load_i2
    shared.dist.init_process_group('nccl', timeout=datetime.timedelta(hours=3))
    assert shared.dist.get_world_size() == 8
    try:
        shared.run_arm('I2', shots, overwrite=False)
        if shared.dist.get_rank() == 0:
            jobs = shared.jobs_for('I2', shots)
            assert len(jobs) == 4888
            # Verify actual decodability, not only filenames or DONE claims.
            for job in jobs:
                with shared.Image.open(job['out']) as im:
                    assert im.size == (96, 96)
                    im.verify()
            shared.atomic_json(shared.OUT / 'I2_MATCHED_VERIFIED.json', dict(
                status='completed', images=len(jobs), shots=list(shots),
                checkpoint=str(ROOT / 'runs/I2-V0915-S3407/global_step_10000'),
                weights='ema.pth', base_evaluator_sha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((shared.OUT / 'PROTOCOL.json').read_bytes()).hexdigest()))
        shared.dist.barrier()
    finally:
        shared.dist.destroy_process_group()


if __name__ == '__main__':
    main()
