"""I data interface: V0913 unchanged main task, frozen donor sets, online ref pixels."""
from h_runtime import *
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scripts.hrfont_i import IModel


def model_for(arm,device,output):
    args = args_for(PARENT0)
    T._verify_caches(args,PARENT0)
    torch.manual_seed(3407)
    base = T.FontDiffuserModel(unet=T.build_unet(args),style_encoder=T.build_style_encoder(args),
                              content_encoder=T.build_content_encoder(args))
    T._load_parent(base,PARENT0,output)
    base.style_encoder.requires_grad_(False).eval()
    base.content_encoder.requires_grad_(False).eval()
    model = IModel(base,arm).to(device)
    stats=torch.load(CACHE/'teacher_stats.pt',map_location=device,weights_only=True)
    model.teacher_mean.copy_(stats['mean']); model.teacher_std.copy_(stats['std'])
    return model,args


class DataContext:
    def __init__(self,args,device):
        self.args,self.device=args,device
        self.es=EsCache(Path(args.es_cache_path)); self.ec=EcCache(Path(args.ec_cache_path))
        donors=load_donors(args.v0913_clean_map); groups=load_cp_group(args.v0913_clean_map)
        self.fonts=donors['all']
        self.exclude=lambda cp: extra_exclude_indices(self.fonts,cp,donors,groups)
        self.library=T._LibraryEs(self.es,self.fonts,T._style_chars_from_cache(self.es))
        self.alpha_cfg=T.DeltaConfig(tau=args.delta_tau,eps_alpha=args.delta_eps_alpha,
            k_max=args.delta_k_max,k_top=args.delta_k_top,mode=args.delta_mode,rng_seed=args.seed)
        self.pool=ThreadPoolExecutor(max_workers=4)
        self.cn_features=None

    def enable_cn(self):
        root=ROOT/'artifacts/i_20260915/cn_content'
        manifest=json.loads((root/'COMPLETE.json').read_text())
        assert manifest['parent_ec_sha256']==sha256_file(PARENT0/'content_encoder.pth')
        assert manifest['features_sha256']==sha256_file(root/'features.pt')
        self.cn_features=torch.load(root/'features.pt',map_location='cpu',weights_only=True)

    def cn_batch(self,ds,count):
        assert self.cn_features is not None
        allowed=sorted(set(Path(p).parent.name for p in ds.target_images))
        rows=[]
        for _ in range(count):
            font=random.choice(allowed)
            cp=random.choice(sorted(ds.style_by_font_char[font]))
            refs=random.sample([c for c in sorted(ds.style_by_font_char[font]) if c!=cp],random.randint(1,8))
            assert cp not in refs and ds.phase=='train'
            target=str(ds.style_by_font_char[font][cp])
            im=self.image(target)
            rows.append(dict(font_stem=font,char_cp=cp,split='train',ref_chars=refs,
                ref_image_paths=[str(ds.style_by_font_char[font][r]) for r in refs],
                target_image=im,nonorm_target_image=(im+1)/2,target_image_path=target,cn_aux=True))
        return batch_to(rows,self.device)

    @staticmethod
    def image(path):
        with Image.open(path) as im:
            assert im.size==(96,96)
            return torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/127.5-1

    @torch.no_grad()
    def conditions(self,samples,cfg,source,no_delta=False):
        style,queries,*_=T._style_conditions(self.es,samples,self.device)
        lengths=[len(x) for x in samples['ref_chars']]
        keep=torch.arange(max(lengths))[None]<torch.tensor(lengths)[:,None]
        paths=sum(samples['ref_image_paths'],[])
        images=list(self.pool.map(self.image,paths))
        refs=torch.zeros(len(lengths),max(lengths),3,96,96)
        refs[keep]=torch.stack(images)
        if 'cn_aux' in samples:
            assert self.cn_features is not None
            content=[torch.cat([self.cn_features[cp][s] for cp in samples['char_cp']]).float().to(self.device) for s in range(5)]
            query=content[-1].clone()
            structure=[(c.new_zeros(len(lengths),1,*c.shape[1:]),c.new_ones(len(lengths),1),c,
                        c.new_zeros(len(lengths))) for c in content]
            return style,refs.pin_memory().to(self.device,non_blocking=True),query,keep.to(self.device),[
                c.masked_fill(cfg[:,None,None,None],0) for c in content],structure
        selections=[]
        for font,cp,chars,q in zip(samples['font_stem'],samples['char_cp'],samples['ref_chars'],queries):
            idx,alpha,_=T.compute_alpha(q.to(self.device),self.library.prototypes(chars,self.device),
                self.library.font_index.get(font),self.alpha_cfg,extra_exclude=self.exclude(cp))
            assert len(idx)>0
            selections.append(([self.fonts[int(i)] for i in idx],alpha))
        neutral=self.ec.features_many('content',[('',c) for c in samples['char_cp']])
        donors=self.ec.features_many('target',[(f,c) for (fonts,_),c in zip(selections,samples['char_cp']) for f in fonts])
        content=[torch.cat([neutral[('',cp)][s] for cp in samples['char_cp']]).to(self.device) for s in range(5)]
        query=content[-1].clone()
        structure=[]
        m=max(len(x[0]) for x in selections)
        alpha=torch.zeros(len(lengths),m,device=self.device)
        for i,(_,a) in enumerate(selections):
            alpha[i,:len(a)]=torch.as_tensor(a,device=self.device)
        for s,c in enumerate(content):
            d=c.new_zeros(len(lengths),m,*c.shape[1:])
            for i,((fonts,_),cp) in enumerate(zip(selections,samples['char_cp'])):
                d[i,:len(fonts)]=torch.cat([donors[(f,cp)][s] for f in fonts]).to(self.device)-c[i]
            structure.append((d,alpha,c,(~(cfg|source)).to(c.dtype)))
        return style,refs.pin_memory().to(self.device,non_blocking=True),query,keep.to(self.device),[
            c.masked_fill(cfg[:,None,None,None],0) for c in content],structure
