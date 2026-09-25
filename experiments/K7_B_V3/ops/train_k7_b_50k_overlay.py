"""Run the immutable K7-B trainer with the separately approved 50K budget.

The original 20K authorization and trainer source remain untouched. This
overlay only redirects the authorization filename so a continuation run can
record a distinct 50K budget and provenance.
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


CODE = Path("/root/projects/hrfont_k7_v3_20260922")
SOURCE_PATH = CODE / "experiments/K6/implementation/r3/scripts/train_k6.py"
RUNTIME_SCRIPTS = SOURCE_PATH.parent
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(RUNTIME_SCRIPTS))
sys.path.insert(0, str(CODE))
identity_link = Path("/root/projects/hrfont/K7_CODE_IDENTITY.json")
if not identity_link.exists():
    identity_link.symlink_to(CODE / "K7_CODE_IDENTITY.json")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
old = "AUTHORIZATION_K7_B.json"
new = "AUTHORIZATION_K7_B_50K.json"
if SOURCE.count(old) != 1:
    raise RuntimeError("K7-B trainer authorization contract changed")
SOURCE = SOURCE.replace(old, new, 1)
old_limit = "assert 1<=a.limit<=20000 and (world==8 or (a.smoke and world==1))"
new_limit = "assert 1<=a.limit<=50000 and (world==8 or (a.smoke and world==1))"
if SOURCE.count(old_limit) != 1:
    raise RuntimeError("K7-B trainer limit contract changed")
SOURCE = SOURCE.replace(old_limit, new_limit, 1)
old_broadcast = "for tensor in raw.state_dict().values():dist.broadcast(tensor,src=0)"
new_broadcast = (
    "dist.barrier() if os.environ.get('K7_SKIP_MODEL_BROADCAST') == '1' "
    "else [dist.broadcast(tensor,src=0) for tensor in raw.state_dict().values()]"
)
if SOURCE.count(old_broadcast) != 1:
    raise RuntimeError("K7-B model synchronization contract changed")
SOURCE = SOURCE.replace(old_broadcast, new_broadcast, 1)
from k5_runtime import K7DynamicEs
import k4_runtime as K4


class K7SharedDynamicEs(K7DynamicEs):
    """Build the immutable dynamic ES cache once and mmap it on every rank."""

    def __init__(self, extra, encoder, device, keys):
        run_id = sys.argv[sys.argv.index("--run-id") + 1]
        cache_dir = Path("/root/projects/hrfont/runs") / run_id / "k7_dynamic_es_shared"
        metadata = cache_dir / "metadata.json"
        spatial_path = cache_dir / "spatial.npy"
        pooled_path = cache_dir / "pooled.npy"
        rank = int(os.environ["RANK"])
        if rank == 0:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not metadata.exists() or not spatial_path.exists() or not pooled_path.exists():
                built = K7DynamicEs(extra, encoder, device, keys)
                with (cache_dir / "spatial.npy.tmp").open("wb") as stream:
                    np.save(stream, built.spatial, allow_pickle=False)
                with (cache_dir / "pooled.npy.tmp").open("wb") as stream:
                    np.save(stream, built.pooled, allow_pickle=False)
                os.replace(cache_dir / "spatial.npy.tmp", spatial_path)
                os.replace(cache_dir / "pooled.npy.tmp", pooled_path)
                metadata.write_text(
                    json.dumps(
                        {
                            "keys": len(built.dynamic_index),
                            "dynamic_entries": built.dynamic_entries,
                            "dynamic_encoder_sha256": built.dynamic_encoder_sha256,
                        }
                    ),
                    encoding="utf-8",
                )
        dist.barrier()
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        super(K7DynamicEs, self).__init__()
        self.old = K4.EsCache(K4.ROOT / "artifacts/g0/es_spatial")
        self.extra = extra
        self.encoder, self.device = encoder, device
        missing = []
        for key in dict.fromkeys(keys):
            font, cp = key.split("|")[2:4]
            if key not in self.old.table.index and (
                extra is None or key not in extra.es.table.index
            ):
                missing.append((font, cp))
        self.spatial = np.load(spatial_path, mmap_mode="r", allow_pickle=False)
        self.pooled = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
        assert self.spatial.shape[0] == len(missing) == meta["keys"]
        self.dynamic_index = {
            K4.key_es("train", font, cp): index
            for index, (font, cp) in enumerate(missing)
        }
        self.dynamic_encoder_sha256 = meta["dynamic_encoder_sha256"]
        self.dynamic_entries = meta["dynamic_entries"]


import k5_runtime as K5_RUNTIME

K5_RUNTIME.K7DynamicEs = K7SharedDynamicEs
old_resume = """    if a.resume:
        assert json.loads((out/'config.json').read_text())==config,'Resume configuration/source changed'
        state=torch.load(a.resume/'trainer.pt',map_location='cpu',weights_only=False)
        assert state['identity']==identity and len(state['rngs'])==world
        raw.load_train_state(torch.load(a.resume/'model.pth',map_location=device,weights_only=True))
        ema=torch.load(a.resume/'ema.pth',map_location=device,weights_only=True)
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step,attempt=state['step'],state['attempt'];exposure.update(state['exposure'])
        assert sum(exposure.values())==64*step
        set_rng(state['rngs'][rank]);del state
"""
new_resume = """    if a.resume:
        assert json.loads((out/'config.json').read_text())==config,'Resume configuration/source changed'
        if os.environ.get('K7_SEQUENTIAL_RESUME') == '1':
            for resume_rank in range(world):
                if rank == resume_rank:
                    state=torch.load(a.resume/'trainer.pt',map_location='cpu',weights_only=False)
                    assert state['identity']==identity and len(state['rngs'])==world
                    raw.load_train_state(torch.load(a.resume/'model.pth',map_location=device,weights_only=True))
                    ema=torch.load(a.resume/'ema.pth',map_location=device,weights_only=True)
                    optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
                    step,attempt=state['step'],state['attempt'];exposure.update(state['exposure'])
                    assert sum(exposure.values())==64*step
                    set_rng(state['rngs'][rank]);del state
                dist.barrier()
        else:
            state=torch.load(a.resume/'trainer.pt',map_location='cpu',weights_only=False)
            assert state['identity']==identity and len(state['rngs'])==world
            raw.load_train_state(torch.load(a.resume/'model.pth',map_location=device,weights_only=True))
            ema=torch.load(a.resume/'ema.pth',map_location=device,weights_only=True)
            optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            step,attempt=state['step'],state['attempt'];exposure.update(state['exposure'])
            assert sum(exposure.values())==64*step
            set_rng(state['rngs'][rank]);del state
"""
if SOURCE.count(old_resume) != 1:
    raise RuntimeError("K7-B resume loading contract changed")
SOURCE = SOURCE.replace(old_resume, new_resume, 1)

exec(
    compile(SOURCE, str(SOURCE_PATH), "exec"),
    {
        "__name__": "__main__",
        "__file__": str(SOURCE_PATH),
        "K7SharedDynamicEs": K7SharedDynamicEs,
    },
)
