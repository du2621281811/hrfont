import unittest
import torch
from e12c_model import Encoder,SetHead,multi_positive
from e12c_metrics import roc_auc_score


class TestE12c(unittest.TestCase):
    def test_auc_ties_and_pairs(self):
        self.assertEqual(roc_auc_score([0,1],[0,1]),1.)
        self.assertEqual(roc_auc_score([0,1],[1,0]),0.)
        self.assertEqual(roc_auc_score([0,1],[1,1]),.5)
        y=[1,0,1,0,1,0];s=[.2,.4,.4,.2,.5,.5]
        expected=sum((s[i]>s[j])+.5*(s[i]==s[j]) for i in range(6) for j in range(6) if y[i]==1 and y[j]==0)/9
        self.assertEqual(roc_auc_score(y,s),expected)

    def test_mask_permutation_duplicate(self):
        torch.manual_seed(1);h=SetHead().eval();q=torch.randn(2,512);r=torch.randn(2,1,512)
        a=h(q,r,torch.ones(2,1,dtype=torch.bool))[0]
        b=h(q,r.repeat(1,8,1),torch.ones(2,8,dtype=torch.bool))[0]
        torch.testing.assert_close(a,b)
        refs=torch.randn(2,8,512);mask=torch.arange(8)[None].expand(2,-1)<3
        p=torch.randperm(8)
        torch.testing.assert_close(h(q,refs,mask)[0],h(q,refs[:,p],mask[:,p])[0])
        refs2=refs.clone();refs2[~mask]=999
        torch.testing.assert_close(h(q,refs,mask)[0],h(q,refs2,mask)[0])
        with self.assertRaises(ValueError):h(q,refs,torch.zeros_like(mask))

    def test_multi_positive_gradient(self):
        x=torch.eye(4).repeat_interleave(2,0).requires_grad_();ids=torch.arange(4).repeat_interleave(2)
        loss=multi_positive(x,x,ids,ids);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertLess(float(loss),1.)
        with self.assertRaises(ValueError):multi_positive(x,x,ids,torch.zeros_like(ids))

    def test_frozen_bn_not_affine(self):
        e=Encoder(pretrained=False).train()
        bn=e.net.bn1;before=bn.running_mean.clone()
        z=e(torch.randn(2,3,96,96));z[:,0].sum().backward()
        self.assertTrue(torch.equal(before,bn.running_mean));self.assertIsNotNone(bn.weight.grad)
        torch.testing.assert_close(z.norm(dim=1),torch.ones(2))


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
