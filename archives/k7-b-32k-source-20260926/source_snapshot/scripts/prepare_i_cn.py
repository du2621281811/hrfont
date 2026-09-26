"""Train-only Han content extension, separate from immutable V0913 main caches."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image, features as pil_features
from i_runtime import ROOT,T,PARENT0,args_for,sha256_file,atomic_json,DataContext
from scripts.build_cn2west_v2_proto_abc import render_glyph_AB

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/i_20260915/cn_content',
                        help='Use a fresh directory for a non-destructive rebuild.')
    opts=parser.parse_args()
    torch.set_num_threads(1)
    assert Image.__version__=='12.2.0' and pil_features.version('freetype2')=='2.14.3', 'use the isolated pinned render_deps PYTHONPATH'
    args=args_for(PARENT0)
    root=opts.out
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
    # Existing western pixels are an audit reference, NOT regenerated training data.
    # Han content has no legacy counterpart: pin it as a separately versioned neutral anchor.
    # Do not claim that today's renderer exactly reproduces the historical PNG tree.
    chars=json.loads((ROOT/'manifests/charset_cn2west_v2_planned.json').read_text())['style_han_338']
    assert len(chars)==338 and len(set(chars))==338
    # CPU preparation keeps all eight GPUs dedicated to the running I1 arm.
    ec=T.build_content_encoder(args).eval().requires_grad_(False)
    ec.load_state_dict(torch.load(PARENT0/'content_encoder.pth',map_location='cpu',weights_only=True))
    features={}
    checks_cn=[]
    with torch.no_grad():
        for ch in chars:
            cp=f'u{ord(ch):04X}'; path=root/f'{cp}.png'
            render_glyph_AB(font,ch,size).save(path)
            gray=np.array(Image.open(path).convert('L'))
            ys,xs=np.where(gray<128)
            assert len(xs)>20 and xs.min()>=5 and ys.min()>=5 and xs.max()<=90 and ys.max()<=90, ('invalid neutral Han',cp)
            checks_cn.append(dict(cp=cp,bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],ink_pixels=len(xs)))
            last,res=ec(DataContext.image(path)[None])
            features[cp]=[x.cpu().half() for x in list(res)+[last]]
    tmp=root/'features.tmp'; torch.save(features,tmp); tmp.replace(root/'features.pt')
    atomic_json(root/'COMPLETE.json',dict(dataset='v0913_clean+I2_CN_Noto83_Pillow12.2',n=338,
        parent_ec_sha256=sha256_file(PARENT0/'content_encoder.pth'),font_sha256=sha256_file(Path(font)),
        features_sha256=sha256_file(root/'features.pt'),content_size=size,renderer_checks=checks,
        pillow=Image.__version__,freetype=pil_features.version('freetype2'),
        source_backend=protocol,cn_geometry_checks=checks_cn,
        neutral_png_sha256={cp:sha256_file(root/f'{cp}.png') for cp in features},
        renderer_note='Not a bitwise reproduction of old western PNG. New Han-only neutral anchors, fixed font/size/backend; original main-task PNGs and caches unchanged.',
        chars=sorted(features),scope='neutral content only; targets/refs from allowed TRAIN fonts, leave target out of refs'))
    print('I_CN_COMPLETE',flush=True)

if __name__=='__main__': main()
