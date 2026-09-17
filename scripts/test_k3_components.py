"""Numerical invariants for the new objective; no pilot training."""
import torch
from scripts.k3_components import pair_loss

def main():
    torch.set_num_threads(1)
    gt=torch.ones(2,3,96,96);gt[0,:,20:70,20:25]=0;gt[1,:,20:70,20:35]=0
    perfect,_=pair_loss(gt,gt);assert perfect.item()==0
    collapsed=gt.mean(0,keepdim=True).repeat(2,1,1,1).requires_grad_()
    loss,parts=pair_loss(collapsed,gt);assert loss>0 and parts['difference']>0
    loss.backward();assert collapsed.grad.abs().sum()>0
    # Swapping both target and prediction order preserves the same objective.
    a,_=pair_loss(collapsed.detach(),gt);b,_=pair_loss(collapsed.detach().flip(0),gt.flip(0))
    assert torch.allclose(a,b)
    # Arbitrary separation is not rewarded: correct GT has strictly lower error.
    wrong,_=pair_loss(gt.flip(0),gt);assert wrong>perfect
    print('K3_OBJECTIVE_INVARIANTS_PASSED')

if __name__=='__main__':main()
