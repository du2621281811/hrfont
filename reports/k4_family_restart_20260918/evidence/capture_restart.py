import datetime,json,os,pathlib,shutil,subprocess,time
C=pathlib.Path('/root/data1/hrfont_k4_family_20260918/control')
R=pathlib.Path('/root/projects/hrfont/runs/K4-A-K1FT-0917-FAMILY-S3407')
read=lambda p:json.loads(p.read_text())
status=read(C/'status.json');h=read(R/'heartbeat.json');config=read(R/'config.json')
assert status['stage']=='K4-A_TRAIN_10000' and status['order']==['K4-A','K4-C','K4-B']
assert h['step']>=100 and h['skips']==0 and h['ddp_spread']==0 and h['global_batch']==64 and h['beta']==.8
assert h['donor_family_guard']['violations']==0 and h['donor_family_guard']['masked_family_candidates']>0
pid=read(C/'QUEUE_PID.json')['pid'];os.kill(pid,0);os.kill(status['pid'],0)
ranks=[]
for rank in range(8):
    rows=[json.loads(l) for l in (R/f'rank{rank}.jsonl').read_text().splitlines()]
    r=rows[-1];assert r['step']>=100 and r['skips']==0 and r['ddp_spread']==0 and r['donor_family_guard']['violations']==0
    ranks.append({k:r[k] for k in ['step','loss','reader_grad','router_grad','grad_norm','beta','global_batch','skips','ddp_spread','donor_family_guard']})
proofs={}
for name in ['CONTRACTS_PASSED','FAMILY_POLICY_PASSED','GPU_PARITY_PASSED','TRAINING_CHECKS_PASSED','PREFLIGHT_PASSED','REPLAY_IDENTITY_PASSED']:
    d=read(C/(name+'.json'));assert d['status']=='PASS';proofs[name]='PASS'
snapshot=dict(captured_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              queue_pid=pid,status=status,heartbeat=h,configuration=config,ranks=ranks,proofs=proofs,
              original_C_stopped=read(pathlib.Path('/root/projects/hrfont/runs/K4-C-K1RECIPE-V2-S3407/STOPPED.json')),
              gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True),
              free_GiB={p:shutil.disk_usage(p).free/2**30 for p in ['/root/data1','/root/projects/hrfont']})
(C/'LIVE_RESTART.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
print(json.dumps(dict(time=snapshot['captured_at'],step=h['step'],queue_pid=pid,loss=h['loss'],donor_guard=h['donor_family_guard'],free=snapshot['free_GiB'])))
