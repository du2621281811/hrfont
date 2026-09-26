"""CPU checks for region supervision and mass-preserving capped sampling."""
import unittest
import torch
from scripts.i56_components import region_masks, region_detail_distance, visual_losses
from scripts.i56_sampling import capped_distribution, sampling_distribution, episode_indices


class TestI56(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407); torch.set_num_threads(1)

    def test_regions_partition_and_outline_interior(self):
        y = torch.ones(1, 1, 96, 96)
        y[:, :, 20:76, 20:76] = 0
        y[:, :, 24:72, 24:72] = 1
        ink, near, bg, _ = region_masks(y)
        torch.testing.assert_close(ink + near + bg, torch.ones_like(y))
        self.assertEqual(float(near[0, 0, 48, 48]), 1)
        self.assertEqual(float(bg[0, 0, 0, 0]), 1)

    def test_identity_blank_solid_empty_regions(self):
        for val in (0., .5, 1.):
            y = torch.full((2, 3, 96, 96), val)
            torch.testing.assert_close(region_detail_distance(y, y), torch.zeros(2))
            p = (1-y).requires_grad_()
            loss = region_detail_distance(p, y).mean(); loss.backward()
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_mask_target_and_alpha(self):
        y = torch.ones(2, 1, 96, 96, requires_grad=True)
        p = torch.zeros_like(y, requires_grad=True)
        tc = torch.zeros_like(y, requires_grad=True)
        loss, c, r = visual_losses(tc, p, y, torch.tensor([True, False]), torch.tensor([.5, .5]), 1000, .2)
        loss.backward()
        self.assertIsNone(y.grad)
        self.assertGreater(float(p.grad[0].abs().sum()), 0)
        self.assertEqual(float(p.grad[1].abs().sum()), 0)
        torch.testing.assert_close(loss.detach(), .02*c.detach()+.2*r.detach())

    def test_hole_and_missing_ink_both_penalized(self):
        y = torch.ones(1, 1, 96, 96); y[:, :, 20:76, 20:76] = 0; y[:, :, 26:70, 26:70] = 1
        filled = y.clone(); filled[:, :, 26:70, 26:70] = 0
        self.assertGreater(float(region_detail_distance(filled, y)), .1)
        self.assertGreater(float(region_detail_distance(torch.ones_like(y), y)), .1)

    def test_sampling_script_caps_and_empty(self):
        base = torch.tensor([.1, .1, .15, .15, .19, .19, .12], dtype=torch.float64)
        groups = [0, 0, 0, 0, 1, 1, 2]; fonts = [0, 0, 1, 1, 2, 3, 4]
        selected = [True, True, False, False, True, False, False]
        late = capped_distribution(base, groups, fonts, selected)
        for g in (0, 1, 2):
            mask = torch.tensor(groups) == g
            torch.testing.assert_close(late[mask].sum(), base[mask].sum())
        self.assertTrue(bool((late <= 3*base+1e-10).all()))
        self.assertGreater(float(late[0]), float(base[0]))
        torch.testing.assert_close(sampling_distribution(base, late, 0), base)
        torch.testing.assert_close(capped_distribution(base, groups, fonts, [False]*7), base)

    def test_tiny_detail_pool_returns_mass(self):
        base = torch.ones(100, dtype=torch.float64)/100
        p = capped_distribution(base, [0]*100, list(range(100)), [True]+[False]*99)
        self.assertAlmostEqual(float(p[0]), .03)
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertTrue(bool((p > 0).all()))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            capped_distribution([.5,.5], [0,0], [0,0], [True,False])
        with self.assertRaises(ValueError):
            region_detail_distance(torch.zeros(1,1,96,96), torch.full((1,1,96,96), 2.))

    def test_fixed_overfit_and_independent_main_rng(self):
        base=torch.ones(10,dtype=torch.float64)/10
        flags=[True]*3+[False]*7
        target=capped_distribution(base,[0]*10,list(range(10)),flags)
        first=episode_indices(base,target,flags,0,0,True)
        self.assertEqual(first,episode_indices(base,target,flags,300,300,True))
        self.assertTrue(all(i<3 for i in first))
        first=episode_indices(base,target,flags,300,300,False)
        torch.randn(100)
        self.assertEqual(first,episode_indices(base,target,flags,300,300,False))


if __name__ == '__main__': unittest.main()
