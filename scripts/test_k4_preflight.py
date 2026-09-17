"""Data, numerical parity, K3-gradient and full-state continuation acceptance."""
import argparse,collections,copy,hashlib,json,random,shutil
from pathlib import Path
import numpy as np
import torch
from k4_runtime import *
from h_runtime import args_for
from k_runtime import DataContext as LegacyContext, dataset as legacy_dataset
from k4_eval import sample_batch,get_sample

OUT=STORE/'control'

def contracts():
    summary={};pools=json.loads((DATA/'manifests/v2/style_pool.json').read_text())
    families=json.loads((ASSETS/'weight_groups.json').read_text())['family_by_font']
    assert set(families)==set(pools)
    source={r['stem']:r['source'] for r in tsv(DATA/'manifests/v2/fonts.tsv')}
    for name,count,nfonts in [('v2',95976,423),('v0917',39547,200)]:
        args=configure(args_for(PARENT0),name,name);ds=dataset(args,'train');assert len(ds)==count
        donors=json.loads((DATA/'manifests'/name/'donor_train_by_cp.json').read_text())
        truth=collections.defaultdict(set)
        for r,p in zip(ds.rows,ds.target_images):
            assert Path(p).is_file();truth[r['cp']].add(r['font'])
        assert {c:set(fs) for c,fs in donors.items()}==dict(truth)
        sampler=KSampler(ds,ASSETS/'detail_manifest.json');exposure=collections.Counter();seen=set();digest=hashlib.sha256()
        for attempt in range(10000):
            ids,quota=sampler.batch(attempt);digest.update(np.asarray(ids,dtype=np.int64).tobytes());seen.update(ids)
            for i in ids:
                f,c,sc,flag=sampler.entries[i];exposure[source[f]+'|'+sc+'|'+str(flag)]+=1
        assert len({sampler.entries[i][0] for i in seen})==nfonts
        assert len({sampler.entries[i][1] for i in seen})==295
        assert sum(exposure.values())==640000
        # The worst-case reference set can remove the incomplete donors only;
        # check every legal query-font/target pair has a fully covered candidate.
        complete={f for f in {r['font'] for r in ds.rows} if len(pools[f])==338}
        minima=[]
        for font,cp in ds.lookup:
            # Complete-CN candidates cover every possible sampled ref set;
            # require a legal candidate after removing the entire query family.
            count=sum(families[d]!=families[font] for d in truth[cp]&complete)
            assert count>0,(name,font,cp)
            minima.append(count)
        for f in {r['font'] for r in ds.rows}:
            assert all(p.is_file() for p in ds.style_by_font_char[f].values())
        summary[name]=dict(pairs=len(ds),fonts=nfonts,unique_pairs_sampled=len(seen),episode_sequence_sha256=digest.hexdigest(),exposure=dict(exposure),minimum_unrelated_complete_donors=min(minima))
    q=[json.loads(l) for l in (ASSETS/'v2_val_test_queries.jsonl').read_text().splitlines()]
    assert len(q)==14419
    for split,n in [('val',7331),('test',7088)]:
        ds=dataset(configure(args_for(PARENT0)),split);qq=[j for j in q if j['split']==split]
        assert len(ds)==len(qq)==n and set(ds.lookup)=={(j['font'],j['cp']) for j in qq}
        for j in qq:assert all(c in ds.style_by_font_char[j['font']] for c in j['ref8'])
    for kind in ['es','ec']:
        c=json.loads((STORE/'cache'/kind/'COMPLETE.json').read_text())
        for n,h in c['files_sha256'].items():assert sha256_file(STORE/'cache'/kind/n)==h
    atomic_json(OUT/'CONTRACTS_PASSED.json',dict(status='PASS',samplers=summary))

def gpu():
    torch.cuda.set_device(0);torch.set_num_threads(1);device=torch.device('cuda:0')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    qs=[json.loads(l) for l in (ASSETS/'v2_val_test_queries.jsonl').read_text().splitlines()]
    oldargs=args_for(PARENT0);old=LegacyContext(oldargs,device)
    newargs=configure(copy.copy(oldargs),'v2','legacy');new=DataContext(newargs,device)
    families=new.library.family_policy.groups
    old_train_families={families[f] for f in old.fonts}
    rows=[]
    for script,k in [('western',1),('western',2),('kana',4),('bopomofo',8)]:
        # Equality is expected only where the new family rule removes no old
        # donor. Family-overlap queries are intentionally changed by this fix.
        j=next(j for j in qs if j['source']=='0913' and j['split']=='val' and j['script']==script and families[j['font']] not in old_train_families)
        rows.append(dict(**j,k=k,refs=j['ref8'][:k]))
    ds=dataset(newargs,'val');samples=[get_sample(ds,j) for j in rows];batch=batch_to(samples,device)
    scheduler=T.build_ddpm_scheduler(oldargs);checks={}
    for arm in ['K0','K1']:
        model,args=model_for(arm,device,OUT/f'parity_{arm}',evaluation=True)
        if arm=='K1':model.load_train_state(torch.load(K1_EMA,map_location=device,weights_only=True))
        model.eval()
        active_rows=[dict(j,k=1,refs=j['ref8'][:1]) for j in rows] if arm=='K0' else rows
        batch=batch_to([get_sample(ds,j) for j in active_rows],device)
        a=sample_batch(model,old,batch,scheduler,[j['seed'] for j in rows])
        b=sample_batch(model,new,batch,scheduler,[j['seed'] for j in rows])
        errors=np.abs(a.astype(float)-b.astype(float)).mean((1,2,3))/255
        assert float(errors.max())<.0002,(arm,errors)
        checks[arm+'_legacy_adapter_mae']=errors.tolist();del model;torch.cuda.empty_cache()
    # K0 reuse also requires byte-identical source files and checkpoint identity.
    for l in (ASSETS/'k0_reuse_candidates.jsonl').read_text().splitlines():
        r=json.loads(l);assert sha256_file(Path(r['path']))==r['sha256']
        root=ROOT/'data/fontdiffuser-p253-t295-s338-cn2west-v2'
        rels=[Path(r['split'])/'TargetImage'/r['font']/f"{r['font']}+{r['cp']}.png",Path(r['split'])/'ContentImage'/f"{r['cp']}.png"]
        rels += [Path(r['split'])/'StyleImage'/r['font']/f"{r['font']}+{c}.png" for c in r['refs']]
        for rel in rels:assert sha256_file(root/rel)==sha256_file(DATA/'v2'/rel)
    del old,new
    model,args=model_for('K4-C',device,OUT/'v2_smoke',evaluation=True);model.eval();data=DataContext(args,device)
    jobs=[]
    for f,sp in [('FZXLB','train'),('ZKTBanQTFU','train'),('ZKTMingXTFU','train'),('FZZJ-MJZCFU','test')]:
        ds=dataset(args,sp);i=next(i for i,r in enumerate(ds.rows) if r['font']==f)
        s=ds[i];jobs.append(s)
        assert all(Path(p).is_file() for p in s['ref_image_paths'])
        assert 'u4E66' not in ds.style_by_font_char[f]
    pp=sample_batch(model,data,batch_to(jobs,device),scheduler,[3407]*4)
    assert pp.shape==(4,96,96,3)
    checks['missing_reference_fonts']=list(['FZXLB','ZKTBanQTFU','ZKTMingXTFU','FZZJ-MJZCFU'])
    # New-bank per-cp exclusion: these fonts must never donate their absent cps.
    for f in ['fzsj_1966717','fzsj_1966721','fzsj_1966722','FZSJ-WULXEM']:
        assert f not in data.library.allowed['u0101']
    del model
    a,args=model_for('K4-A',device,OUT/'warm_start_A')
    state=torch.load(K1_EMA,map_location=device,weights_only=True)
    assert all(torch.equal(v,state[k]) for k,v in a.train_state().items())
    checks['K4_A_complete_EMA_warmstart']=True
    atomic_json(OUT/'GPU_PARITY_PASSED.json',dict(status='PASS',checks=checks,k0_reuse_approved=True))

def training_checks():
    ref=ROOT/'runs/K4-WEIGHT-PREFLIGHT-C';res=ROOT/'runs/K4-WEIGHT-PREFLIGHT-C-RESUME';b=ROOT/'runs/K4-WEIGHT-PREFLIGHT-B';a=ROOT/'runs/K4-WEIGHT-PREFLIGHT-A'
    logs=lambda p:{r['step']:r for r in (json.loads(l) for l in (p/'train_log.jsonl').read_text().splitlines())}
    x,y,z,w=map(logs,[ref,res,b,a])
    assert max(x)==max(y)==120 and max(z)==17 and max(w)==2
    for log in [x,y,z,w]:assert all(r['ddp_spread']<1e-4 and r['skips']==0 and r['global_batch']==64 for r in log.values())
    for log in [x,y,z,w]:assert all(r['donor_family_guard']['calls']>0 and r['donor_family_guard']['violations']==0 and r['donor_family_guard']['minimum_eligible']>0 for r in log.values())
    errors={}
    for step in [101,102]:
        err=abs(x[step]['loss']-y[step]['loss']);assert err<1e-5,(step,err)
        assert x[step]['episode_sha256']==y[step]['episode_sha256'];errors[step]=err
    auxiliary_startup=[]
    for rank in range(8):
        xx={r['step']:r for r in (json.loads(l) for l in (ref/f'rank{rank}.jsonl').read_text().splitlines())}
        zz={r['step']:r for r in (json.loads(l) for l in (b/f'rank{rank}.jsonl').read_text().splitlines())}
        assert all(xx[s]['episode_sha256']==zz[s]['episode_sha256'] for s in range(1,18))
        assert all(v>0 for v in zz[1]['rollout_epsilon_grad_norms'])
        assert zz[1]['pair_reader_grad']>0
        # F0 introduces zero-initialized structural residual convolutions. At
        # update 1 the image loss cannot yet reach the upstream router. Check
        # the next auxiliary event, using the global gradient aggregation used
        # by the trainer; individual FP16 ranks can still underflow this early.
        assert zz[1]['pair_router_grad']==0
        assert np.isfinite(zz[17]['pair_router_grad']) and zz[17]['pair_router_grad']>=0
        auxiliary_startup.append(dict(rank=rank,first_router_grad=zz[1]['pair_router_grad'],
                                     next_router_grad=zz[17]['pair_router_grad']))
    assert sum(r['next_router_grad']**2 for r in auxiliary_startup)>0
    assert all(r['beta']==.8 for r in w.values())
    ss=[torch.load(p/'last_state/model.pth',map_location='cpu',weights_only=True) for p in [ref,res]]
    delta=sum((ss[0][k].float()-ss[1][k].float()).square().sum().item() for k in ss[0])
    norm=sum(v.float().square().sum().item() for v in ss[0].values());relative=(delta/max(norm,1e-20))**.5
    assert relative<.001,relative
    # Val192 path actually exercised on the resume smoke's final update.
    assert json.loads((res/'eval_step_120/DONE.json').read_text())['rows']==192
    checks=dict(status='PASS',first_resume_losses=errors,final_relative_state_l2=relative,
                synchronized_ranks=8,amp_skips=0,same_B_C_main_episode_sequences=True,
                K3_rollout_gradient=True,auxiliary_router_startup=auxiliary_startup)
    # This restart must run its own preflight with the actual family-filtered
    # runtime, not reuse the former target-font-only training checks.
    current=code_identity()
    for directory in [ref,res,b,a]:
        original=json.loads((directory/'config.json').read_text())['identity']
        assert original==current
    atomic_json(OUT/'TRAINING_CHECKS_PASSED.json',checks)
    parity=json.loads((OUT/'GPU_PARITY_PASSED.json').read_text());contracts=json.loads((OUT/'CONTRACTS_PASSED.json').read_text())
    assert parity['status']==contracts['status']=='PASS'
    assert json.loads((OUT/'FAMILY_POLICY_PASSED.json').read_text())['status']=='PASS'
    atomic_json(OUT/'PREFLIGHT_PASSED.json',dict(status='PASS',identity=code_identity(),k0_reuse_approved=parity['k0_reuse_approved'],training=checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contracts','gpu','training_checks']);a=p.parse_args()
    {'contracts':contracts,'gpu':gpu,'training_checks':training_checks}[a.mode]()
