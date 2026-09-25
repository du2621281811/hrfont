import json,csv,collections,pathlib,math
R=pathlib.Path('outputs/K5_EVALUATION_20260919');O=R/'statistics';O.mkdir(exist_ok=True)
metrics=['l1','ssim','D_region','D_change','D_add','D_remove','D_high','D_out']; data={};summary=[];paired=[]
difficulty={}
f=R/'gt_difficulty/difficulty_font_script.csv'
if f.exists():
 for r in csv.DictReader(f.open()):difficulty[r['split'],r['font'],r['script']]=r['difficulty']
for arm in ['K5-A','K5-B']:
 data[arm]={};groups=collections.defaultdict(list)
 for split in ['train','val','test']:
  for r in json.loads((R/arm/split/'metrics.json').read_text())['rows']:
   key=(split,r['font'],r['cp'],r['k']);assert key not in data[arm];data[arm][key]=r
   script=r['script'];dif=difficulty.get((split,r['font'],script),difficulty.get((split,r['font'],'latin' if script=='western' else script),'unknown'));r['difficulty']=dif
   for m in metrics:assert math.isfinite(r[m])
   for dims in [(split,'all','all','all'),(split,str(r['k']),'all','all'),(split,'all',script,'all'),(split,'all','all',dif),(split,str(r['k']),script,'all'),(split,str(r['k']),script,dif)]:groups[dims].append(r)
 for dims,rs in groups.items():
  v=dict(model=arm,split=dims[0],shot=dims[1],script=dims[2],difficulty=dims[3],n=len(rs),fonts=len(set(r['font'] for r in rs)))
  fonts=collections.defaultdict(list)
  for r in rs:fonts[r['font']].append(r)
  for m in metrics:
   v[m]=sum(r[m] for r in rs)/len(rs);v[m+'_font_macro']=sum(sum(r[m] for r in rr)/len(rr) for rr in fonts.values())/len(fonts)
  summary.append(v)
assert data['K5-A'].keys()==data['K5-B'].keys()
for k,a in data['K5-A'].items():
 b=data['K5-B'][k];assert all(a[x]==b[x] for x in ['refs','seed','target','content'])
 paired.append(dict(split=k[0],font=k[1],cp=k[2],shot=k[3],script=a['script'],difficulty=a['difficulty'],**{m+'_B_minus_A':b[m]-a[m] for m in metrics}))
def write(name,rows):
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('STRATIFIED.csv',summary);write('PAIRED_PER_IMAGE.csv',paired)
lookup={(r['model'],r['split'],r['shot'],r['script'],r['difficulty']):r for r in summary};delta=[]
for r in summary:
 if r['model']!='K5-A':continue
 b=lookup['K5-B',r['split'],r['shot'],r['script'],r['difficulty']]
 delta.append(dict(split=r['split'],shot=r['shot'],script=r['script'],difficulty=r['difficulty'],n=r['n'],**{m+'_B_minus_A':b[m]-r[m] for m in metrics}))
write('PAIRED_SUMMARY.csv',delta)
lines=['# K5-A/B 自动指标统计','', '每组59980张；A/B逐图核对split/font/字符/shot、参考字、seed、GT及content路径一致。L1与SSIM为最终PNG对GT的指标，不等于训练loss。D_*为现有代码的局部/重绘误差，除SSIM越高越好外，其余越低越好。','', 'Train是旧默认固定查询子集；val/test是完整V2冻结查询。不同split不作直接难度比较。CSV同时提供逐图均值及字体等权宏均值，shot统计含重复字符的不同参考数结果。','', '| Split | Shot | N/model | A L1 | B L1 | A SSIM | B SSIM | A D_out | B D_out |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
 if r['model']!='K5-A' or r['script']!='all' or r['difficulty']!='all':continue
 b=lookup['K5-B',r['split'],r['shot'],'all','all'];lines.append(f"| {r['split']} | {r['shot']} | {r['n']} | {r['l1']:.5f} | {b['l1']:.5f} | {r['ssim']:.5f} | {b['ssim']:.5f} | {r['D_out']:.5f} | {b['D_out']:.5f} |")
lines+=['','难度使用GT相对中性内容图的几何重绘需求，阈值仅以0913训练集校准；不是人评难度。未知标签保留unknown，不强行归类。','', 'D_change：GT要求增加/移除笔画区域误差；D_add/D_remove分别统计两类区域；D_high：高通局部误差；D_region：现有局部区域误差；D_out = D_region + D_change + 0.25 D_high。无区域时的零值按现有实现纳入，跨script解读需谨慎。','', '此表不包含尚未计算的LPIPS/FID或E12-c，不能用L1/SSIM代替空心、端点、纹理的视觉验收。']
lines += ['', '## 分script汇总', '', '| Split | Script | A L1 | B L1 | A SSIM | B SSIM |', '|---|---|---:|---:|---:|---:|']
for r in summary:
 if r['model']=='K5-A' and r['shot']=='all' and r['script']!='all' and r['difficulty']=='all':
  b=lookup['K5-B',r['split'],'all',r['script'],'all'];lines.append(f"| {r['split']} | {r['script']} | {r['l1']:.5f} | {b['l1']:.5f} | {r['ssim']:.5f} | {b['ssim']:.5f} |")
lines += ['', '## 分GT难度汇总', '', '| Split | Difficulty | A L1 | B L1 | A SSIM | B SSIM |', '|---|---|---:|---:|---:|---:|']
for r in summary:
 if r['model']=='K5-A' and r['shot']=='all' and r['script']=='all' and r['difficulty']!='all':
  b=lookup['K5-B',r['split'],'all','all',r['difficulty']];lines.append(f"| {r['split']} | {r['difficulty']} | {r['l1']:.5f} | {b['l1']:.5f} | {r['ssim']:.5f} | {b['ssim']:.5f} |")
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
