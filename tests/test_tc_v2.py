"""CPU contract tests for the TC-v2 implementation."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
sys.path.insert(0, str(VARIANT))

from src.model import _add_tc_global_residual  # noqa: E402
from src.tc_v2 import (TC_APPEARANCE_DIM, TCV2Cache, TCV2Global9Adapter,
                       TCV2Head)


class TCV2Contracts(unittest.TestCase):
  def test_head_gradient_mask_and_teacher_inputs(self):
    torch.manual_seed(3407)
    head = TCV2Head()
    refs = torch.randn(2, 3, TC_APPEARANCE_DIM)
    ec = torch.randn(2, 707)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    out = head(refs, ec, mask)
    assert out.shape == (2, TC_APPEARANCE_DIM)
    out.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and
               float(p.grad.abs().sum()) > 0 for p in head.parameters())
    with self.assertRaisesRegex(ValueError, "at least one valid"):
        head(refs, ec, torch.zeros_like(mask))


  def test_zero_init_shared_projection_and_global9_injection(self):
    adapter = TCV2Global9Adapter()
    appearance = torch.randn(2, TC_APPEARANCE_DIM)
    residual = adapter(appearance)
    assert residual.shape == (2, 9, 1024)
    assert torch.equal(residual, torch.zeros_like(residual))
    seq = torch.randn(2, 12, 1024)
    zero = _add_tc_global_residual(seq, residual)
    assert torch.equal(zero, seq)
    changed = _add_tc_global_residual(seq, torch.ones_like(residual))
    assert torch.equal(changed[:, :9], seq[:, :9] + 1)
    assert torch.equal(changed[:, 9:], seq[:, 9:])


  def test_cache_manifest_and_split_binding(self):
    import tempfile
    tmp_path = Path(tempfile.mkdtemp(prefix="tc-v2-test-"))
    keys = ["tc|ref|train|fontA|u4E00", "tc|target|train|fontA|u4E01"]
    (tmp_path / "keys.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")
    arr = np.memmap(tmp_path / "appearance.dat", dtype=np.float16, mode="w+",
                    shape=(2, TC_APPEARANCE_DIM))
    arr[:] = 0
    arr.flush()
    stats = {"mean": [0.0] * TC_APPEARANCE_DIM, "scale": [1.0] * TC_APPEARANCE_DIM}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "kind": "tc_v2_appearance_cache", "appearance_dim": 896,
        "dtype": "float16", "entries": 2,
        "train_target_stats": stats,
    }), encoding="utf-8")
    (tmp_path / "progress.json").write_text(json.dumps({"done": 2, "total": 2}), encoding="utf-8")
    cache = TCV2Cache(tmp_path)
    cache.validate_split("train")
    assert cache.stats("ref", "train", "fontA", "u4E00").shape == (896,)
    cache.validate_split("test")  # test refs may be used at inference
    with self.assertRaises(ValueError):
        cache.validate_split("dev")
    bad = Path(tempfile.mkdtemp(prefix="tc-v2-bad-"))
    (bad / "keys.txt").write_text(keys[0] + "\n", encoding="utf-8")
    np.memmap(bad / "appearance.dat", dtype=np.float16, mode="w+",
              shape=(1, 1)).flush()
    (bad / "manifest.json").write_text(json.dumps({"kind": "wrong", "appearance_dim": 896}),
                                        encoding="utf-8")
    (bad / "progress.json").write_text(json.dumps({"done": 1, "total": 1}), encoding="utf-8")
    with self.assertRaisesRegex(RuntimeError, "manifest kind"):
        TCV2Cache(bad)


if __name__ == "__main__":
    unittest.main()
