import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hrfont_ref_aggregation import RefGlobal9Reader, apply_reader, attach_reader

spec = importlib.util.spec_from_file_location('ref_attention', ROOT / 'code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/attention.py')
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


class RefTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.r = RefGlobal9Reader(channels=16, hidden=16, heads=4)
        self.x = torch.randn(2, 4, 16, 3, 3)
        self.keep = torch.ones(2, 4, dtype=torch.bool)

    def test_zero_init_and_gradients(self):
        torch.testing.assert_close(self.r(self.x, self.keep), self.x.mean(1))
        optim = torch.optim.AdamW(self.r.parameters(), lr=1e-3)
        for _ in range(2):
            optim.zero_grad()
            self.r(self.x, self.keep).square().mean().backward()
            self.assertGreater(self.r.out.weight.grad.norm().item(), 0)
            optim.step()
        self.assertGreater(self.r.key.weight.grad.norm().item(), 0)
        self.assertGreater(self.r.queries.grad.norm().item(), 0)

    def test_permutation_duplicate_padding(self):
        torch.nn.init.normal_(self.r.out.weight, std=.01)
        y = self.r(self.x, self.keep)
        torch.testing.assert_close(y, self.r(self.x[:, [2, 0, 3, 1]], self.keep))
        torch.testing.assert_close(y, self.r(self.x.repeat(1, 2, 1, 1, 1), self.keep.repeat(1, 2)))
        x = torch.cat([self.x, torch.randn_like(self.x)], 1)
        keep = torch.cat([self.keep, torch.zeros_like(self.keep)], 1)
        torch.testing.assert_close(y, self.r(x, keep))

    def test_cfg_and_rng(self):
        holder = torch.nn.Module()
        before = torch.get_rng_state().clone()
        attach_reader(holder)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        holder.ref_reader = self.r
        torch.nn.init.normal_(self.r.out.bias)
        out = apply_reader(holder, self.x, {'ref_chars': [[1]*4, [1]*4]}, torch.tensor([True, False]))
        self.assertEqual(out[0].abs().sum().item(), 0)
        self.assertGreater(out[1].abs().sum().item(), 0)

    def test_count_norm_duplication_and_slice(self):
        a = attention.CrossAttention(16, 16, heads=4, dim_head=4)
        q = torch.randn(2, 3, 16)
        g, loc = torch.randn(2, 9, 16), torch.randn(2, 16, 16)
        c1, c8 = torch.cat([g, loc], 1), torch.cat([g, loc.repeat(1, 8, 1)], 1)
        m1, m8 = torch.ones(2, 25, dtype=torch.bool), torch.ones(2, 137, dtype=torch.bool)
        old1 = a(q, c1, m1)
        a.local_count_norm = True
        torch.testing.assert_close(old1, a(q, c1, m1))
        torch.testing.assert_close(a(q, c1, m1), a(q, c8, m8))
        full = a(q, c8, m8)
        a._slice_size = 2
        torch.testing.assert_close(full, a(q, c8, m8))
        m8[:, 9:] = False
        self.assertTrue(torch.isfinite(a(q, c8, m8)).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
