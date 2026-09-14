import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.hrfont_h import AppearanceMemory, lr_factor


class HTests(unittest.TestCase):
    def test_reader_invariances_and_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(3407)
        m = AppearanceMemory()
        r = torch.randn(2, 1, 144, 256)
        q = torch.randn(2, 256, 12, 12)
        a = m(r, q, torch.ones(2, 1, dtype=torch.bool))[0]
        b = m(r.expand(-1, 8, -1, -1), q, torch.ones(2, 8, dtype=torch.bool))[0]
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        r = torch.randn(2, 4, 144, 256)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        a, pred, _ = m(r, q, mask)
        perm = torch.tensor([3, 0, 2, 1])
        b = m(r[:, perm], q, mask[:, perm])[0]
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        r[~mask] = float('nan')
        torch.testing.assert_close(a, m(r, q, mask)[0])
        (a.square().mean() + pred.square().mean()).backward()
        for n, p in m.named_parameters():
            self.assertIsNotNone(p.grad, n)
            self.assertTrue(torch.isfinite(p.grad).all(), n)
            self.assertGreater(float(p.grad.norm()), 0, n)

    def test_lr(self):
        self.assertAlmostEqual(lr_factor(500), 1)
        self.assertAlmostEqual(lr_factor(5000), 1)
        self.assertAlmostEqual(lr_factor(10000), .1)
        self.assertGreater(lr_factor(7500), lr_factor(9000))


if __name__ == '__main__':
    unittest.main()
