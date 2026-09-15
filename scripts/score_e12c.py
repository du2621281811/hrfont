"""Score generated query + supplied Chinese refs, without target GT or font label."""
import argparse
import json
from pathlib import Path
import torch
from e12c_model import Encoder,SetHead
from train_e12c import Images
from new_data_inventory import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--query',type=Path,required=True);ap.add_argument('--refs',type=Path,nargs='+',required=True)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if not 1<=len(a.refs)<=8:raise ValueError('Supply 1 to 8 actual reference images')
    done=json.loads((a.checkpoint/'DONE.json').read_text());assert not done['smoke']
    assert sha(a.checkpoint/'A_best.pt')==done['encoder_sha256']
    assert sha(a.checkpoint/'B_best.pt')==done['head_sha256']
    device=torch.device(a.device);e=Encoder(False).to(device);h=SetHead().to(device)
    e.load_state_dict(torch.load(a.checkpoint/'A_best.pt',map_location=device,weights_only=False)['model'])
    h.load_state_dict(torch.load(a.checkpoint/'B_best.pt',map_location=device,weights_only=False)['head'])
    e.eval();h.eval();data=Images([{'path':str(p)} for p in [a.query]+a.refs])
    with torch.no_grad():
        z=e(torch.stack([data[i] for i in range(len(data))]).to(device))
        score,cos=h(z[:1],z[1:][None],torch.ones(1,len(a.refs),device=device,dtype=torch.bool))
    print(json.dumps(dict(logit=float(score),cosine_diagnostic=float(cos),shots=len(a.refs),
        encoder_sha256=done['encoder_sha256'],head_sha256=done['head_sha256'])))


if __name__=='__main__':main()
