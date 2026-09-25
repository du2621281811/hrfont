#!/usr/bin/env python3
"""Small CPU integration probes for the TC-v2 train-side helper.

This test intentionally extracts only the helper functions from ``train.py`` so
it does not require the full CUDA/diffusers training runtime.  The fake caches
use the production feature shapes and exercise the same cache-to-head path.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
TRAIN = VARIANT / "train.py"


def load_tc_module():
    path = VARIANT / "src/tc_v2.py"
    spec = importlib.util.spec_from_file_location("tc_v2_review", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_conditions_helper():
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    names = {"_pool_ec", "_tc_conditions"}
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__",
                             names=[ast.alias(name="annotations")], level=0)] + funcs,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {"torch": torch, "F": F}
    exec(compile(module, str(TRAIN), "exec"), scope)
    return scope["_tc_conditions"]


def load_builder_collect_jobs():
    path = ROOT / "scripts/build_tc_v2_cache.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and node.name in {"char_map", "collect_jobs"}]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__",
                             names=[ast.alias(name="annotations")], level=0)] + funcs,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {"Path": Path}
    exec(compile(module, str(path), "exec"), scope)
    return scope["collect_jobs"]


def load_checkpoint_helpers():
    path = VARIANT / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_load_checkpoint", "_load_weights_only"}
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__",
                             names=[ast.alias(name="annotations")], level=0)] + funcs,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {"torch": torch, "Path": Path, "json": json}
    exec(compile(module, str(path), "exec"), scope)
    return scope["_load_checkpoint"], scope["_load_weights_only"]


def load_sampler_parse_args():
    path = ROOT / "scripts/sample_tc_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    config_path = VARIANT / "configs/fontdiffuser.py"
    config_spec = importlib.util.spec_from_file_location("review_fontdiffuser_config", config_path)
    config = importlib.util.module_from_spec(config_spec)
    assert config_spec.loader is not None
    config_spec.loader.exec_module(config)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__",
                             names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {"sys": sys, "Path": Path, "get_parser": config.get_parser}
    exec(compile(module, str(path), "exec"), scope)
    return scope["parse_args"]


class FakeAppearanceCache:
    def __init__(self, tc):
        self.tc = tc

    def validate_split(self, split):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)

    def stats(self, role, split, font, cp):
        # Make target and reference rows distinct, while retaining [896].
        value = 1.0 if role == "target" else 0.25
        return torch.full((self.tc.TC_APPEARANCE_DIM,), value)


class FakeEcCache:
    def features(self, role, font, cp):
        channels = (3, 64, 128, 256, 256)
        return [torch.ones(1, c, 2, 2) for c in channels]


class TCV2IntegrationReview(unittest.TestCase):
    def test_helper_shape_cfg_mask_and_gradients(self):
        tc = load_tc_module()
        helper = load_conditions_helper()
        head = tc.TCV2Head()
        adapter = tc.TCV2Global9Adapter()
        samples = {
            "split": ["train", "train"],
            "font_stem": ["font-a", "font-b"],
            "char_cp": ["u0031", "u0032"],
            "ref_chars": [["u5929"], ["u548C", "u5730"]],
        }
        cfg_mask = torch.tensor([True, False])
        residual, comp_loss, predicted = helper(
            FakeAppearanceCache(tc), FakeEcCache(), samples, head, adapter,
            cfg_mask, "cpu", include_target=True,
        )
        self.assertEqual(tuple(predicted.shape), (2, tc.TC_APPEARANCE_DIM))
        self.assertEqual(tuple(residual.shape), (2, tc.TC_GLOBAL_TOKENS, tc.TC_GLOBAL_DIM))
        self.assertIsNotNone(comp_loss)
        self.assertTrue(torch.equal(residual[0], torch.zeros_like(residual[0])))
        # The adapter is zero-initialized, so ``residual.square()`` would have
        # a mathematically zero first derivative at step 0.  Use a non-zero
        # probe target to verify that W_out can actually receive a gradient.
        adapter_probe = (residual[1] - 1.0).square().mean()
        (comp_loss + adapter_probe).backward()
        self.assertIsNotNone(head.mlp[-1].weight.grad)
        self.assertIsNotNone(adapter.proj.weight.grad)
        self.assertGreater(float(adapter.proj.weight.grad.abs().sum()), 0.0)

    def test_head_rejects_all_masked_references(self):
        tc = load_tc_module()
        head = tc.TCV2Head()
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 2, tc.TC_APPEARANCE_DIM),
                 torch.zeros(1, tc.TC_EC_DIM),
                 torch.zeros(1, 2, dtype=torch.bool))

    def test_builder_clean_map_excludes_dirty_and_test_targets(self):
        """The production job enumerator must never cache dirty/test teachers."""
        collect_jobs = load_builder_collect_jobs()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            clean = Path(td) / "clean"
            font = "FZDemo"
            split = {"train": [font], "val": [font], "test": [font]}
            for part in split:
                for kind in ("StyleImage", "TargetImage"):
                    directory = root / part / kind / font
                    directory.mkdir(parents=True)
                    # The collector only resolves names; the encoder is tested
                    # separately against real PNGs.
                    (directory / f"{font}+u0030.png").touch()
                    (directory / f"{font}+u0031.png").touch()
            clean.mkdir()
            for part in ("train", "val", "test"):
                (clean / f"pairs_{part}.tsv").write_text(
                    "font\tcp\n" + f"{font}\tu0030\n", encoding="utf-8")
            jobs = collect_jobs(root, split, clean)
            target_jobs = {(part, cp) for role, part, _font, cp, _path in jobs
                           if role == "target"}
            self.assertEqual(target_jobs, {("train", "u0030"), ("val", "u0030")})
            self.assertFalse(any(role == "target" and part == "test"
                                 for role, part, *_rest in jobs))

    def test_cache_constructor_rejects_malformed_manifest(self):
        """A consumer must fail closed before mmap-ing a forged cache."""
        tc = load_tc_module()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            (directory / "keys.txt").write_text(
                "tc|ref|train|F|u0030\n", encoding="utf-8")
            array = __import__("numpy").memmap(
                directory / "appearance.dat", dtype="float16", mode="w+",
                shape=(1, tc.TC_APPEARANCE_DIM))
            array[:] = 0
            array.flush()
            (directory / "manifest.json").write_text(json.dumps({
                "kind": "wrong", "appearance_dim": 1,
                "train_target_stats": {"mean": [0] * tc.TC_APPEARANCE_DIM,
                                        "scale": [1] * tc.TC_APPEARANCE_DIM},
            }), encoding="utf-8")
            with self.assertRaises((RuntimeError, ValueError, KeyError)):
                tc.TCV2Cache(directory)

    def test_legacy_sampling_call_matches_pipeline_signature(self):
        """Never leave an online TC kwarg after reverting legacy DPM changes."""
        sample_path = VARIANT / "sample.py"
        pipeline_path = VARIANT / "src/dpm_solver/pipeline_dpm_solver.py"
        sample = ast.parse(sample_path.read_text(encoding="utf-8"))
        pipeline = ast.parse(pipeline_path.read_text(encoding="utf-8"))
        generate = next(node for node in ast.walk(pipeline)
                        if isinstance(node, ast.FunctionDef) and node.name == "generate")
        accepted = {arg.arg for arg in generate.args.args}
        calls = [node for node in ast.walk(sample)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "generate"]
        self.assertTrue(calls)
        for call in calls:
            self.assertTrue({kw.arg for kw in call.keywords if kw.arg} <= accepted)

    def test_warm_start_does_not_restore_trainer_step_but_resume_does(self):
        """The two load paths must remain observably different."""
        load_checkpoint, load_weights_only = load_checkpoint_helpers()
        module = torch.nn.Linear(2, 2)
        original = {name: value.detach().clone()
                    for name, value in module.state_dict().items()}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            for name in ("unet", "style_encoder", "content_encoder"):
                torch.save(original, directory / f"{name}.pth")
            torch.save({"step": 37}, directory / "trainer_state.pt")
            # The loader only requires these attributes; sharing a tiny module
            # keeps this a CPU-only contract test.
            model = SimpleNamespace(unet=module, style_encoder=module,
                                    content_encoder=module)
            self.assertIsNone(load_weights_only(model, directory))
            self.assertEqual(load_checkpoint(model, directory), 37)

    def test_tc_resume_rejects_empty_module_state(self):
        """An empty placeholder must not silently randomize a TC resume."""
        load_checkpoint, load_weights_only = load_checkpoint_helpers()
        module = torch.nn.Linear(2, 2)
        tc_head = torch.nn.Linear(2, 2)
        tc_adapter = torch.nn.Linear(2, 2)
        original = {name: value.detach().clone()
                    for name, value in module.state_dict().items()}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            for name in ("unet", "style_encoder", "content_encoder"):
                torch.save(original, directory / f"{name}.pth")
            torch.save({}, directory / "tc_head.pth")
            torch.save({}, directory / "tc_global_adapter.pth")
            torch.save({"step": 1}, directory / "trainer_state.pt")
            model = SimpleNamespace(unet=module, style_encoder=module,
                                    content_encoder=module, tc_head=tc_head,
                                    tc_global_adapter=tc_adapter)
            with self.assertRaises((RuntimeError, ValueError, KeyError)):
                load_checkpoint(model, directory, require_tc=True)

    def test_tc_resume_binding_mismatch_rejects_and_match_loads(self):
        """TC resume must bind H/W weights to the exact cache/Ec lineage."""
        load_checkpoint, load_weights_only = load_checkpoint_helpers()
        module = torch.nn.Linear(2, 2)
        tc_head = torch.nn.Linear(2, 2)
        tc_adapter = torch.nn.Linear(2, 2)
        base = {name: value.detach().clone() for name, value in module.state_dict().items()}
        binding = {"tc_cache_manifest_sha256": "tc-sha",
                   "ec_checkpoint_sha256": "ec-sha"}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            for name in ("unet", "style_encoder", "content_encoder"):
                torch.save(base, directory / f"{name}.pth")
            torch.save(tc_head.state_dict(), directory / "tc_head.pth")
            torch.save(tc_adapter.state_dict(), directory / "tc_global_adapter.pth")
            (directory / "tc_binding.json").write_text(json.dumps(binding), encoding="utf-8")
            torch.save({"step": 4}, directory / "trainer_state.pt")
            model = SimpleNamespace(unet=module, style_encoder=module,
                                    content_encoder=module, tc_head=tc_head,
                                    tc_global_adapter=tc_adapter, tc_binding=binding)
            self.assertEqual(load_checkpoint(model, directory, require_tc=True), 4)
            load_weights_only(model, directory)
            model.tc_binding = {**binding, "ec_checkpoint_sha256": "wrong"}
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, directory, require_tc=True)
            with self.assertRaises(RuntimeError):
                load_weights_only(model, directory)

    def test_tc_sampler_diffusion_loop_is_inference_only(self):
        """The expensive cond/uncond loop must not retain autograd graphs."""
        path = ROOT / "scripts/sample_tc_v2.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        loop = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id == "timestep")
        no_grad = [node for node in ast.walk(tree)
                   if isinstance(node, ast.With)
                   and any(isinstance(item.context_expr, ast.Call)
                           and ((isinstance(item.context_expr.func, ast.Attribute)
                                 and item.context_expr.func.attr in {"no_grad", "inference_mode"})
                                or (isinstance(item.context_expr.func, ast.Name)
                                    and item.context_expr.func.id in {"no_grad", "inference_mode"}))
                           for item in node.items)]
        main = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        decorated = any(isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr in {"no_grad", "inference_mode"}
                        for dec in main.decorator_list)
        self.assertTrue(decorated or any(any(loop in ast.walk(parent) for parent in node.body)
                                         for node in no_grad))

    def test_tc_sampler_parse_supports_f2_baseline_without_tc(self):
        parse_args = load_sampler_parse_args()
        old_argv = sys.argv
        try:
            for arm in ("F2", "F2RL"):
                sys.argv = ["sample_tc_v2.py", "--arm", arm,
                            "--no-tc_enabled", "--checkpoint", "/tmp/ckpt",
                            "--data-root", "/tmp/data", "--font", "F",
                            "--content-cp", "u0030", "--refs", "u5929",
                            "--output", "/tmp/out"]
                args = parse_args()
                self.assertEqual(args.arm, arm)
                self.assertFalse(args.tc_enabled)
        finally:
            sys.argv = old_argv

    def test_optimizer_groups_gate_local_lr_and_clip_all_trainables(self):
        """Default local LR keeps one base group; explicit LR opts into split."""
        path = VARIANT / "train.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        local_group_ifs = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            text = ast.unparse(node.test)
            if "local_trainable" in text and "local_learning_rate" in text:
                local_group_ifs.append(node)
        separate_assignments = [node for node in ast.walk(tree)
                                if isinstance(node, ast.Assign)
                                and any(isinstance(target, ast.Name)
                                        and target.id == "separate_local"
                                        for target in node.targets)]
        separate_text = " ".join(ast.unparse(node.value) for node in separate_assignments)
        self.assertTrue(any("is not None" in ast.unparse(node.test)
                            for node in local_group_ifs)
                        or ("local_trainable" in separate_text
                            and "local_learning_rate" in separate_text
                            and "is not None" in separate_text))
        # Every optimizer parameter group must be represented by the clipping
        # set; otherwise F2RL local weights silently bypass max_grad_norm.
        clip_calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "clip_grad_norm_"]
        self.assertTrue(clip_calls)
        clip_text = " ".join(ast.unparse(node) for node in clip_calls)
        clip_params = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name)
                               and target.id == "clip_params" for target in node.targets)]
        clip_text += " " + " ".join(ast.unparse(node.value) for node in clip_params)
        self.assertIn("local_trainable", clip_text)

    def test_head_adapter_tiny_cpu_overfit(self):
        """A tiny synthetic target must drive both H and the zero-init W_out."""
        tc = load_tc_module()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(3407)
            head, adapter = tc.TCV2Head(), tc.TCV2Global9Adapter()
            optimizer = torch.optim.AdamW(list(head.parameters()) + list(adapter.parameters()),
                                          lr=3e-3)
            refs = torch.randn(4, 2, tc.TC_APPEARANCE_DIM)
            ec = torch.randn(4, tc.TC_EC_DIM)
            target = torch.tanh(torch.randn(4, tc.TC_APPEARANCE_DIM))
            target_tokens = torch.randn(4, tc.TC_GLOBAL_TOKENS, tc.TC_GLOBAL_DIM)
            mask = torch.ones(4, 2, dtype=torch.bool)
            predicted = head(refs, ec, mask)
            residual = adapter(predicted)
            initial = F.smooth_l1_loss(predicted, target) + 0.01 * F.mse_loss(
                residual, target_tokens)
            initial.backward()
            self.assertIsNotNone(head.mlp[-1].weight.grad)
            self.assertIsNotNone(adapter.proj.weight.grad)
            self.assertGreater(float(adapter.proj.weight.grad.abs().sum()), 0.0)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(30):
                predicted = head(refs, ec, mask)
                residual = adapter(predicted)
                loss = F.smooth_l1_loss(predicted, target) + 0.01 * F.mse_loss(
                    residual, target_tokens)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            self.assertLess(losses[-1], losses[0] * 0.5)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
