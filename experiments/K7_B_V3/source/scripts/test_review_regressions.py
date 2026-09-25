#!/usr/bin/env python3
"""CPU regressions for the 2026-09-08 F3/E12 review fixes.

Run: python scripts/test_review_regressions.py
No model downloads, external fonts, or generation training required.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval_framework"))
from scripts.hrfont_support_adapter import SupportAdapter
from data import MembershipDataset
from models import PhiS2
from train_membership import main as train_membership, membership_splits
from train_utils import binary_auc, pr_auc

TRAIN = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser/train.py"


class ReviewRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_support_helper_updates_adapter(self):
        tree = ast.parse(TRAIN.read_text())
        # `_support_tokens` now routes style rows through the production
        # `_pooled_style_cached` helper.  Keep this AST fixture limited to the
        # real helper chain rather than importing the training module (which
        # would pull in optional diffusers/runtime dependencies).
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"_pool_ec", "_pooled_style_cached", "_support_tokens"}]
        scope = {"torch": torch, "_SUPPORT_POOL_CACHE": {}}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
            + helpers,
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, str(TRAIN), "exec"), scope)
        cache = SimpleNamespace(features=lambda *args: [torch.ones(1, 3, 2, 2)])
        adapter = SupportAdapter(3, 4)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.01)
        samples = {"font_stem": ["font"], "char_cp": ["A"]}
        before = adapter.net[-2].weight.detach().clone()
        support = scope["_support_tokens"](
            cache, adapter, {"A": ["ref"]}, samples, [False],
            SimpleNamespace(support_k=1), "cpu")
        self.assertTrue(support.requires_grad)
        target = torch.tensor([[[1., -1., .5, -.5]]])
        (support-target).square().mean().backward()
        self.assertIsNotNone(adapter.net[-2].weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, adapter.net[-2].weight))

    def test_training_support_call_is_outside_no_grad(self):
        tree = ast.parse(TRAIN.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_support_tokens"]
        self.assertTrue(calls)
        for node in ast.walk(main):
            if isinstance(node, ast.With) and any("no_grad" in ast.unparse(i.context_expr) for i in node.items):
                self.assertFalse(any(call in list(ast.walk(node)) for call in calls))

    def test_auc_ties_and_order(self):
        self.assertEqual(binary_auc([1, 1, 0, 0], [0., 0., 0., 0.]), .5)
        self.assertEqual(binary_auc([0, 0, 1, 1], [0., 0., 0., 0.]), .5)
        labels = torch.tensor([1, 0, 1, 0, 1, 0])
        scores = torch.tensor([.2, .2, .7, .1, .7, .7])
        pos, neg = scores[labels == 1], scores[labels == 0]
        expected = ((pos[:, None] > neg).float() + .5*(pos[:, None] == neg)).mean()
        self.assertAlmostEqual(binary_auc(labels, scores), float(expected))

    def test_average_precision_ties(self):
        self.assertEqual(pr_auc([1, 1, 0, 0], [0., 0., 0., 0.]), .5)
        self.assertEqual(pr_auc([0, 0, 1, 1], [0., 0., 0., 0.]), .5)
        self.assertEqual(pr_auc([1, 0], [1., 0.]), 1.)

    def test_membership_pairing_covers_families_and_characters(self):
        ds = MembershipDataset.__new__(MembershipDataset)
        ds.seed = 3407
        ds.families = ["f0", "f1", "f2", "f3"]
        ds.query_chars = ["A", "B"]
        ds.ref_chars = ["R"]
        ds.near = {}
        keys = [(f, c) for f in ds.families for c in ["A", "B", "R"]]
        ds.by = {key: i for i, key in enumerate(keys)}
        ds.base = [(torch.full((3, 2, 2), float(i)), {}) for i in range(len(keys))]
        for pair in range(8):
            positive, negative = ds[2*pair], ds[2*pair+1]
            family = ds.families[pair % 4]
            char = ds.query_chars[pair // 4]
            self.assertEqual(positive[3:], (family, family))
            self.assertEqual(negative[4], family)
            self.assertNotEqual(negative[3], family)
            self.assertEqual(float(positive[2]), 1.)
            self.assertEqual(float(negative[2]), 0.)
            self.assertEqual(float(negative[0][0, 0, 0]), ds.by[(negative[3], char)])

    def test_formal_split_and_explicit_smoke(self):
        manifest = {"fonts": [{"family": f"family{i}", "group": f"g{i}"} for i in range(8)]}
        splits, overlap = membership_splits(manifest, [.5, .25, .25], 3407)
        self.assertFalse(overlap)
        self.assertFalse(set(splits["val"]) & set(splits["test"]))
        tiny = {"fonts": manifest["fonts"][:3]}
        with self.assertRaises(ValueError):
            membership_splits(tiny, [.5, .25, .25], 3407)
        _, overlap = membership_splits(tiny, [.5, .25, .25], 3407, smoke=True)
        self.assertTrue(overlap)

    def test_membership_end_to_end_calibration_and_bn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            records, fonts = [], []
            for i in range(8):
                family = f"family{i}"
                fonts.append({"family": family, "group": family})
                for char in ["A", "R"]:
                    name = f"{family}_{char}.png"
                    Image.new("RGB", (32, 32), (30+i*20, 80, 150)).save(cache / name)
                    records.append({"family": family, "char": char, "path": name})
            (cache / "manifest.json").write_text(json.dumps({"fonts": fonts, "records": records}))
            encoder = PhiS2(feature_dim=8)
            original_bn = {k: v.clone() for k, v in encoder.state_dict().items()
                           if "running_" in k or "num_batches_tracked" in k}
            torch.save({"model": encoder.state_dict()}, root / "phi.pt")
            cfg = {
                "data": {"cache_dir": str(cache), "split_ratios": [.5, .25, .25],
                         "split_seed": 3407, "query_chars": ["A"], "ref_chars": ["R"]},
                "model": {"phi_checkpoint": str(root / "phi.pt"), "in_channels": 3,
                          "feature_dim": 8, "hidden_dim": 8},
                "train": {"seed": 3407, "device": "cpu", "smoke_mode": False,
                          "episodes": 8, "batch_size": 4, "lr": .001,
                          "weight_decay": .01, "max_steps": 2, "epochs": 1},
                "eval": {"episodes": 8}, "calibration": {"method": "temperature"},
                "output": {"dir": str(root / "out")}}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(cfg))
            with contextlib.redirect_stdout(io.StringIO()):
                train_membership(["--config", str(path)])
            val = json.loads((root / "out/val_metrics.json").read_text())
            test = json.loads((root / "out/test_metrics.json").read_text())
            self.assertFalse(set(val["families"]) & set(test["families"]))
            self.assertEqual(val["temperature"], test["temperature"])
            self.assertEqual(test["role"], "held_out_test")
            payload = torch.load(root / "out/best.pt", weights_only=False)
            for key, tensor in original_bn.items():
                self.assertTrue(torch.equal(tensor, payload["model"]["encoder."+key]), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
