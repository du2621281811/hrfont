"""Train-only Han content extension, separate from immutable V0913 main caches."""
import json
import torch
import numpy as np
from PIL import Image, features as pil_features
from i_runtime import ROOT,T,PARENT0,args_for,sha256_file,atomic_json,DataContext
from scripts.build_cn2west_v2_proto_abc import render_glyph_AB

def main():
    torch.set_num_threads(1)
    args=args_for(PARENT0)
    root=ROOT/'artifacts/i_20260915/cn_content'
    if (root/'COMPLETE.json').exists():
        print('CN extension already complete'); return
    root.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/fontdiffuser-p253-t295-s338-cn2west-v2'
    protocol=json.loads((data/'summary.json').read_text())['protocol']
    font=protocol['content_font']; size=protocol['content_size']
    assert protocol['canvas']==96 and size==83
    # Reproduce existing neutral raster before extending the same protocol to Han.
    checks=[]
    for existing in sorted((data/'train/ContentImage').glob('*.png')):
        cp=existing.stem
        expected=np.array(Image.open(data/'train/ContentImage'/f'{cp}.png').convert('L')).astype(int)
        actual=np.array(render_glyph_AB(font,chr(int(cp[1:],16)),size).convert('L')).astype(int)
        error=np.abs(expected-actual)
        mask_a,mask_b=expected<128,actual<128
        iou=float((mask_a&mask_b).sum()/max(1,(mask_a|mask_b).sum()))
        row=dict(cp=cp,max_error=int(error.max()),mean_error=float(error.mean()),binary_iou=iou)
        checks.append(row)
        # Pinned font/size with minor FreeType antialias drift accepted only in this new extension.
        assert row['mean_error']<=.3 and iou>=.98, ('renderer geometry mismatch',row)
    chars=json.loads((ROOT/'manifests/charset_cn2west_v2_planned.json').read_text())['style_han_338']
    assert len(chars)==338 and len(set(chars))==338
    # CPU preparation keeps all eight GPUs dedicated to the running I1 arm.
    ec=T.build_content_encoder(args).eval().requires_grad_(False)
    ec.load_state_dict(torch.load(PARENT0/'content_encoder.pth',map_location='cpu',weights_only=True))
    features={}
    with torch.no_grad():
        for ch in chars:
            cp=f'u{ord(ch):04X}'; path=root/f'{cp}.png'
            render_glyph_AB(font,ch,size).save(path)
            last,res=ec(DataContext.image(path)[None])
            features[cp]=[x.cpu().half() for x in list(res)+[last]]
    tmp=root/'features.tmp'; torch.save(features,tmp); tmp.replace(root/'features.pt')
    atomic_json(root/'COMPLETE.json',dict(dataset='v0913_clean+I2_CN_content',n=338,
        parent_ec_sha256=sha256_file(PARENT0/'content_encoder.pth'),font_sha256=sha256_file(font),
        features_sha256=sha256_file(root/'features.pt'),content_size=size,renderer_checks=checks,
        pillow=Image.__version__,freetype=pil_features.version('freetype2'),
        source_backend=protocol,antialias_note='main V0913 PNG unchanged; CN extension uses recorded runtime FreeType',
        chars=sorted(features),scope='neutral content only; targets/refs from allowed TRAIN fonts, leave target out of refs'))
    print('I_CN_COMPLETE',flush=True)

if __name__=='__main__': main()
