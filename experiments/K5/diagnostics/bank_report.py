import json,pathlib,hashlib,collections,csv,html
p=pathlib.Path('outputs/K5_BANK_DIAGNOSTIC_20260919'); assets=json.loads(pathlib.Path('outputs/K5_EVALUATION_20260919/ASSET_PATHS.json').read_text())
rows=[]; bad=[]
for f in sorted(p.glob('K5-*/case*/*/result.json')):
 d=json.loads(f.read_text()); im=f.parent/'prediction.png'
 if hashlib.sha256(im.read_bytes()).hexdigest()!=d['prediction_sha256']:bad.append(str(im))
 d['local']=str(im.relative_to(p));rows.append(d)
assert len(rows)==576,(len(rows),bad)
assert not bad,bad
modes=['top10','top10_uniform','top50','all','all_uniform','off']; groups=collections.defaultdict(list)
for r in rows:groups[(r['arm'],r['font'],r['mode'])].append(r)
metrics=['l1','ssim','D_out'];out=[]
for (a,f,m),rr in sorted(groups.items()):
 base=groups[a,f,'top10'];out.append(dict(arm=a,font=f,mode=m,n=len(rr),**{k:sum(x[k] for x in rr)/len(rr) for k in metrics},**{'delta_'+k:sum(x[k] for x in rr)/len(rr)-sum(x[k] for x in base)/len(base) for k in metrics}))
with (p/'SUMMARY.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(p/'AUDIT.json').write_text(json.dumps({'count':len(rows),'sha256_failures':bad,'counts':dict(collections.Counter(x['arm'] for x in rows))},indent=2))
cases=collections.defaultdict(dict)
for r in rows: cases[r['arm'],r['case'],r['seed']][r['mode']]=r
body=[]
for (a,c,s),rr in sorted(cases.items()):
 r=rr['top10']; target='assets/'+r['target'].lstrip('/')
 refs=['assets/'+x.lstrip('/') for x in r['refs_paths']]
 pics='<figure><img src="'+target+'"><figcaption>GT</figcaption></figure>'
 for m in modes:
  x=rr[m];pics+=f'<figure><img src="{x["local"]}"><figcaption>{m}<br>L1 {x["l1"]:.4f}</figcaption></figure>'
 body.append(f'<section data-arm="{a}" data-font="{r["font"]}"><h3>{a} · {r["font"]} · {r["cp"]} · case {c} · seed {s}</h3><div class="pics">{pics}</div><details><summary>固定参考字符</summary>'+''.join(f'<img width="96" src="{x}">' for x in refs)+'</details></section>')
(p/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>K5 Bank 六组诊断</title><style>body{font:16px system-ui;background:#eee;margin:30px}section{background:white;padding:18px;margin:18px 0}.pics{display:flex;gap:12px}figure{margin:0;text-align:center}figure img{width:144px;image-rendering:auto}figcaption{font-size:13px}select{padding:8px}</style><h1>K5 Bank 六组诊断</h1><p>576/576 输出及 SHA256 已核验。全部条件为 FP32；不与标准 AMP 推理直接混比。普通字体 case 存在配对重复。L1 不是训练 loss；视觉与路由归因尚待完成。</p><select id="arm"><option value="">全部模型</option><option>K5-A</option><option>K5-B</option></select><select id="font"><option value="">全部字体</option><option>FZBangSKLTJW</option><option>FZPANGPBJW</option><option>FZPTYJW</option></select>'''+''.join(body)+'''<script>function filter(){document.querySelectorAll('section').forEach(x=>x.hidden=(arm.value&&x.dataset.arm!==arm.value)||(font.value&&x.dataset.font!==font.value))}arm.onchange=font.onchange=filter;</script>''')
print(json.dumps(out,ensure_ascii=False,indent=2))
