"""CPU-only unit checks; not GPU, real-data, or end-to-end training acceptance."""
import unittest
import torch
from i34_components import InkReadout, detail_distance, raw_x0, sampling_distribution, visual_losses


class ComponentsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        torch.set_num_threads(1)

    def test_patch_coordinates(self):
        x = torch.arange(144*64).reshape(1, 144, 64)
        y = InkReadout.unpatchify(x)
        self.assertEqual(y.shape, (1, 1, 96, 96))
        for r, c, dy, dx in ((0, 0, 0, 0), (2, 5, 3, 7), (11, 11, 7, 7)):
            self.assertEqual(y[0, 0, r*8+dy, c*8+dx], x[0, r*12+c, dy*8+dx])

    def test_actual_token_projection_receives_gradient(self):
        source = torch.randn(2, 144, 256, requires_grad=True)
        out = torch.nn.Linear(256, 1024)
        read = InkReadout()
        pred = read(out(source))
        target = torch.ones(2, 3, 96, 96)
        target[:, :, 20:70, 40:50] = 0
        detail_distance(pred, target).mean().backward()
        self.assertEqual(sum(p.numel() for p in read.parameters()), 65600)
        for grad in (source.grad, out.weight.grad, read.proj.weight.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_identity_white_and_ink(self):
        y = torch.ones(2, 1, 96, 96)
        y[1, :, 20:70, 30:60] = .25
        self.assertTrue(torch.equal(detail_distance(y, y), torch.zeros(2)))

    def test_target_detached_and_unconditional_mask(self):
        y = torch.ones(2, 1, 96, 96, requires_grad=True)
        p = torch.zeros(2, 1, 96, 96, requires_grad=True)
        loss, _, _ = visual_losses(p, p, y, torch.tensor([True, False]), torch.tensor([.5, .5]), 1000)
        loss.backward()
        self.assertIsNone(y.grad)
        self.assertGreater(float(p.grad[0].abs().sum()), 0)
        self.assertEqual(float(p.grad[1].abs().sum()), 0)

    def test_unclipped_x0_and_schedule_weight(self):
        x = torch.full((2, 3, 96, 96), 2., requires_grad=True)
        eps = torch.zeros_like(x)
        y, weight = raw_x0(x, eps, torch.tensor([.25, 1.]), torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(y[:, 0, 0, 0], torch.tensor([2.5, 1.5])))
        self.assertTrue(torch.equal(weight, torch.tensor([.25, 1.])))
        y.sum().backward()
        self.assertGreater(float(x.grad.min()), 0)

    def test_sampling_preserves_scripts_and_ramps(self):
        base = torch.tensor([.2, .3, .1, .28, .12], dtype=torch.float64)
        groups = torch.tensor([0, 0, 1, 1, 2])
        difficulty = [1, 2, 1.5, 1, 2]
        early = sampling_distribution(base, groups, difficulty, 1000)
        late = sampling_distribution(base, groups, difficulty, 2000)
        self.assertTrue(torch.allclose(early, base))
        self.assertGreater(late[1], base[1])
        self.assertTrue(torch.all(late >= .5*base))
        for g in groups.unique():
            self.assertAlmostEqual(float(late[groups == g].sum()), float(base[groups == g].sum()))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            sampling_distribution([1.], [0], [float('nan')], 2000)
        with self.assertRaises(ValueError):
            detail_distance(torch.zeros(1, 1, 96, 96), torch.full((1, 1, 96, 96), 2.))


if __name__ == '__main__':
    unittest.main()
