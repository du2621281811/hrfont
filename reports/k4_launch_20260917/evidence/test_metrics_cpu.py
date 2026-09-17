import json,pathlib,torch,lpips,hashlib
from scripts.e12c_r2.model import Encoder
from k4_runtime import read_unit,atomic_json,STORE,CODE
contract=json.loads((CODE/'K4_POSTPROCESS_DEPENDENCIES.json').read_text())
for path,h in contract['files'].items():assert hashlib.sha256((CODE/path).read_bytes()).hexdigest()==h
torch.set_num_threads(2)
root=pathlib.Path('/root/data1/hrfont_dataset_v2_20260917/v2')
paths=[root/'train/TargetImage/FZBGDT'/f'FZBGDT+{c}.png' for c in ['u0041','u3042','u3105']]
x=torch.stack([read_unit(p) for p in paths]);net=lpips.LPIPS(net='alex',version='0.1').eval();encoder=Encoder(False,False)
encoder.load_state_dict(torch.load('/root/data1/hrfont_e12c_r2_20260917/R1/A_best.pt',map_location='cpu',weights_only=False)['model']);encoder.eval()
with torch.no_grad():
 d=net(x*2-1,x*2-1)
 z=encoder((x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None])
 assert torch.isfinite(z).all() and float(d.abs().max())<1e-7
 assert torch.allclose(z.norm(dim=1),torch.ones(len(x)),atol=1e-5)
atomic_json(STORE/'control/METRICS_PREFLIGHT.json',dict(status='PASS',samples=len(paths),lpips_self_distance=d.flatten().tolist(),e12_embedding_norms=z.norm(dim=1).tolist(),device='cpu',dependency_contract=contract,purpose='Validate installed frozen evaluator/LPIPS and native96 image normalization'))
print('METRICS_PREFLIGHT_PASS')
