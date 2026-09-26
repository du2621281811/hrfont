import torch
from data import STORE,atomic
errors={}
for stage,key in [('A_best.pt','model'),('B_best.pt','head')]:
    a=torch.load(STORE/'SMOKE_R1'/stage,map_location='cpu',weights_only=False)[key]
    b=torch.load(STORE/'SMOKE_RESUME_R1'/stage,map_location='cpu',weights_only=False)[key]
    assert a.keys()==b.keys();worst=0.
    for k in a:
        assert torch.allclose(a[k],b[k],atol=1e-6,rtol=1e-5),(stage,k)
        worst=max(worst,float((a[k].float()-b[k].float()).abs().max()))
    errors[stage]=worst
atomic(STORE/'PREFLIGHT_PASSED.json',dict(status='PASSED',resume_max_abs=errors,arms=['SMOKE_R1','SMOKE_R2','SMOKE_R3']))
print(errors)
