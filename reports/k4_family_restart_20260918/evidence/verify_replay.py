import sys,json,random,hashlib
sys.path.insert(0,'scripts')
from k4_runtime import *
from h_runtime import args_for
torch.set_num_threads(1);torch.cuda.set_device(0)
ds=dataset(configure(args_for(PARENT0)),'train');sampler=KSampler(ds,ASSETS/'detail_manifest.json')
pools=json.loads((DATA/'manifests/v2/style_pool.json').read_text());checks=[]
for attempt in range(2):
    ids,_=sampler.batch(attempt)
    for rank in range(8):
        seed_episode(3407+attempt*1000+rank);fonts=[];cps=[];refs=[]
        for i in ids[rank*8:rank*8+8]:
            f,cp,_,_=sampler.entries[i];pool=sorted(pools[f]);random.sample(pool,random.randint(1,8))
            fonts.append(f);cps.append(cp);refs.append(random.sample(sorted(set(pool)-{cp}),random.randint(1,8)))
        source=torch.rand(8,device='cuda')<.05;cfg=torch.rand(8,device='cuda')<.02
        noise=torch.randn((8,3,96,96),device='cuda');ts=torch.randint(0,1000,(8,),device='cuda')
        digest=hashlib.sha256(json.dumps([fonts,cps,refs]).encode()+noise.cpu().numpy().tobytes()+ts.cpu().numpy().tobytes()+cfg.cpu().numpy().tobytes()+source.cpu().numpy().tobytes()).hexdigest()
        log=[json.loads(l) for l in (ROOT/'runs/K4-C-K1RECIPE-V2-S3407'/f'rank{rank}.jsonl').read_text().splitlines()]
        assert log[attempt]['episode_sha256']==digest,(attempt,rank)
        checks.append(dict(update=attempt+1,rank=rank,sha256=digest))
out=Path('/root/data1/hrfont_k4_family_20260918/control/REPLAY_IDENTITY_PASSED.json')
atomic_json(out,dict(status='PASS',checks=checks,scope='All eight ranks, first two executed updates: exact font/char/refs/noise/timestep/CFG/source identity.'))
print('REPLAY_IDENTITY_PASS',len(checks))
