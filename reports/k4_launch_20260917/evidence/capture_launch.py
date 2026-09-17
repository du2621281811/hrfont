import datetime,json,pathlib,shutil,subprocess,os
R=pathlib.Path('/root/projects/hrfont/runs/K4-C-K1RECIPE-V2-S3407')
C=pathlib.Path('/root/data1/hrfont_k4_20260917/control')
load=lambda p:json.loads(p.read_text())
h=load(R/'heartbeat.json');conf=load(R/'config.json');status=load(C/'status.json')
assert status['stage']=='K4-C_TRAIN_10000' and h['state']=='TRAINING'
assert h['step']>=100 and h['skips']==0 and h['ddp_spread']==0 and h['global_batch']==64
ranks=[]
for rank in range(8):
    rr=[json.loads(l) for l in (R/f'rank{rank}.jsonl').read_text().splitlines()]
    last=rr[-1];assert last['step']>=100 and last['skips']==0 and last['ddp_spread']==0
    ranks.append({k:last[k] for k in ['step','attempt','loss','reader_grad','router_grad','grad_norm','skips','ddp_spread','global_batch','peak_mib']})
pid=load(C/'QUEUE_PID.json')['pid'];os.kill(pid,0);os.kill(status['pid'],0)
logs=[json.loads(l) for l in (R/'train_log.jsonl').read_text().splitlines()]
base=[json.loads(l) for l in (pathlib.Path('/root/projects/hrfont/runs/K4-PREFLIGHT-C/train_log.jsonl')).read_text().splitlines()]
first=[dict(step=i+1,loss_difference=abs(logs[i]['loss']-base[i]['loss']),same_episode=logs[i]['episode_sha256']==base[i]['episode_sha256']) for i in range(2)]
assert all(x['same_episode'] and x['loss_difference']<1e-5 for x in first)
e=dict(captured_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
       queue_pid=pid,queue_status=status,heartbeat=h,rank_records=ranks,
       execution_commit=conf['identity']['commit'],configuration={k:v for k,v in conf.items() if k!='identity'},
       first_formal_vs_smoke=first,
       disk_free_GiB={p:shutil.disk_usage(p).free/2**30 for p in ['/root/data1','/root/projects/hrfont']},
       gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True),
       gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),
       pending_stages=['K4-B_TRAIN_10000','K4-A_TRAIN_10000','K0_V2_INFERENCE','K1_V2_INFERENCE','K3_V2_INFERENCE','K4-C_V2_INFERENCE','K4-B_V2_INFERENCE','K4-A_V2_INFERENCE','V2_FROZEN_DIFFICULTY','ALL_MODEL_METRICS','REPORT'])
p=C/'LIVE_LAUNCH.json';p.write_text(json.dumps(e,ensure_ascii=False,indent=2));print(json.dumps(dict(time=e['captured_at'],step=h['step'],elapsed=h['elapsed_seconds'],queue_pid=pid,child_pid=status['pid'],disk=e['disk_free_GiB'])))
