"""Behavioral checks for leakage grouping, gradients, set scoring and ranking."""
import itertools,json,unittest
import numpy as np
import torch
import torch.nn.functional as F
from data import grouped
from model import Encoder,SetHead,cosine,multi_positive
from evaluate import rank_metrics,tau_b

class Tests(unittest.TestCase):
    def test_alias_groups_transitive(self):
        rows=[]
        for font in ['train_alias','val_alias','independent']:
            for i in range(8):rows.append(dict(font=font,role='target',script='latin',cp=str(i),rgb_sha256=str(i) if font!='independent' else 'x'+str(i)))
        units,_=grouped(rows,{})
        self.assertEqual(units['train_alias'],units['val_alias']);self.assertNotEqual(units['train_alias'],units['independent'])
    def test_multiscale_masked_cosine(self):
        torch.manual_seed(1)
        q=torch.cat([F.normalize(torch.randn(2,d),dim=1)*w for d,w in [(512,.5**.5),(128,.5),(128,.5)]],1)
        refs=q[:,None,:].repeat(1,8,1);keep=torch.ones(2,8,dtype=torch.bool);refs[:,7]=1000;keep[:,7]=False
        self.assertTrue(torch.allclose(cosine(q,refs,keep),torch.ones(2),atol=1e-6))
    def test_residual_starts_at_cosine_and_set_invariance(self):
        torch.manual_seed(2);h=SetHead(768,True);q=F.normalize(torch.randn(2,768),dim=1);r=F.normalize(torch.randn(2,8,768),dim=2);k=torch.ones(2,8,dtype=torch.bool)
        y,c=h(q,r,k);self.assertTrue(torch.equal(y,c));perm=torch.randperm(8)
        self.assertTrue(torch.allclose(h(q,r[:,perm],k[:,perm])[0],y,atol=1e-6))
        one=h(q,r[:,:1],k[:,:1])[0];rep=h(q,r[:,:1].repeat(1,8,1),k)[0]
        self.assertTrue(torch.allclose(one,rep,atol=1e-6))
    def test_frozen_encoder_retains_input_gradient(self):
        e=Encoder(True,False).eval().requires_grad_(False);x=torch.randn(2,3,96,96,requires_grad=True);z=e(x)
        self.assertEqual(z.shape,(2,768));z[:,0].sum().backward();self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in e.parameters()))
    def test_multi_positive_gradient_and_bn(self):
        e=Encoder(True,False).train();self.assertTrue(all(not m.training for m in e.modules() if isinstance(m,torch.nn.BatchNorm2d)))
        a=torch.randn(4,768,requires_grad=True);b=torch.randn(4,768,requires_grad=True);i=torch.tensor([0,0,1,1])
        loss=multi_positive(a,b,i,i);loss.backward();self.assertTrue(torch.isfinite(a.grad).all())
    def test_exact_original_metric_and_null(self):
        count=0
        for order in itertools.permutations('AGNFX'):
            s={k:float(5-i) for i,k in enumerate(order)};count+=rank_metrics(s)['M5']
        self.assertEqual(count,8)
        s={'A':1.,'G':.8,'N':.5,'F':.1,'X':.9};r=rank_metrics(s)
        self.assertEqual(r['M5'],0);self.assertEqual(r['same_char_top2'],1)
        self.assertIsNone(tau_b([3,2,1,3],[1,1,1,1]));self.assertEqual(rank_metrics(dict.fromkeys('AGNFX',1.))['M5'],0.)
    def test_partial_candidates_not_success(self):
        s={'A':1.,'G':.9,'N':None,'F':.2,'X':.95};r=rank_metrics(s)
        self.assertIsNone(r['M5']);self.assertIsNone(r['N_gt_F']);self.assertEqual(r['X_gt_F'],1.)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
