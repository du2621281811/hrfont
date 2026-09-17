"""Reproducible matched K statistics and E12 GT rank correlation, without GPUs."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ['family_s01','lpips_alex','l1','ssim','D_region','D_change','D_high']
KEY = ['protocol','split','font','cp','k']
BOOT = 10000
SEED = 20260917


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ci(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim==1: matrix=matrix[:,None]
    if len(matrix)<2: return np.full((2,matrix.shape[1]),np.nan)
    rng=np.random.default_rng(SEED)
    boots=matrix[rng.integers(0,len(matrix),(BOOT,len(matrix)))].mean(1)
    return np.quantile(boots,[.025,.975],axis=0)


def slices(frame, base, dimensions):
    for n in range(len(dimensions)+1):
        for extra in itertools.combinations(dimensions,n):
            keys=base+list(extra)
            for values, group in frame.groupby(keys,sort=True,dropna=False):
                if not isinstance(values,tuple): values=(values,)
                labels={d:'all' for d in dimensions}
                labels.update(zip(keys,values))
                yield labels,group


def tau_b(grades,scores):
    con=dis=tx=ty=0
    for i in range(len(grades)):
        for j in range(i):
            x=np.sign(grades[i]-grades[j]); d=scores[i]-scores[j]
            y=0 if abs(d)<=1e-6 else np.sign(d)
            if x and y:
                if x==y: con+=1
                else: dis+=1
            elif not x and y: tx+=1
            elif x and not y: ty+=1
    den=math.sqrt((con+dis+tx)*(con+dis+ty))
    return (con-dis)/den if den else np.nan


def dashboard(out,means,deltas,correlations):
    payload={'means':json.loads(means.to_json(orient='records')),
             'deltas':json.loads(deltas.to_json(orient='records')),
             'correlations':json.loads(correlations.to_json(orient='records'))}
    page='''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>HRFont 分层统计与排序相关性</title><style>
body{font:15px system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#1d2939;background:#f4f6f8}
h1{font-size:28px}h2{font-size:21px;margin-top:30px}p{line-height:1.8}a{color:#165987}label{display:inline-block;margin:8px 16px 8px 0}select{display:block;margin-top:6px;padding:8px;border:1px solid #bfcbd5;border-radius:4px;background:white;min-width:120px}
.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border-bottom:1px solid #dee4e9;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}.note{background:#fff3d9;border-left:4px solid #b58027;padding:12px 18px}.files a{display:inline-block;margin:8px}.muted{color:#596b79}
</style><h1>HRFont · shot × script × 难度</h1>
<p><a href="../index.html">统计总览</a> · <a href="../k_results/index.html">K族图像与逐图分数</a> · <a href="REPORT.md">定义与结论</a></p>
<p>各表在当前分组内先按字体平均，再让字体等权。K0/K1 保持相同目标、参考、噪声与推理配置；Train 用于拟合诊断。原始47字符与默认128字符协议分开。</p>
<div class="note">难度指 GT 相对标准字的重绘需求，按训练字体校准为低 / 中 / 高，随后固定应用到 Val/Test。它不使用模型结果，也不等同于人工风格复杂度。分组少于5个字体时需谨慎解读；所有置信区间都是探索性的逐项区间。</div>
<div id="filters"></div><p id="count" class="muted"></p>
<h2>模型分层均值</h2><div class="scroll" id="means"></div>
<h2>相对 K0 的配对差与95%区间</h2><p class="muted">按字体配对重采样10,000次；正负方向由指标决定。多个小分组未经多重比较校正，不能仅挑显著结果。</p><div class="scroll" id="delta"></div>
<h2>E12-c 与 GT 关系的排序相关性</h2><p>当前 script、难度筛选作用于外部原始测试的 GT 查询。GT 分数与生成器 shot 无关，因此每个字体×字符只计一次，不按 K0/K1 或 1/2/4/8-shot 重复统计。分组等级为 A、X 同级，高于 N，高于 F；Pred 没有预设真值名次。</p><div class="scroll" id="corr"></div>
<p class="note">这里的 Spearman ρ / Kendall τ-b 验证与构造的 GT 关系是否一致，不能替代人评相关性。Pearson r 需要有意义的数值真值；本轮是序数关系，因此不把任意等级编码的 Pearson r 当作核心证据。</p>
<h2>下载完整结果</h2><div class="files">DOWNLOADS</div>
<script>const data=PAYLOAD;
const state={protocol:'original47',split:'test',script:'all',k:'all',difficulty:'all'};
const names={protocol:'推理协议',split:'数据划分',script:'目标 script',k:'生成 shot',difficulty:'GT重绘难度'};
const labels={all:'全部',easy:'低',medium:'中',hard:'高',unknown:'未确定',western:'西文',kana:'假名',bopomofo:'注音'};
const values={protocol:[...new Set(data.means.map(r=>r.protocol))],split:['train','val','test'],script:['all','western','kana','bopomofo'],k:['all',1,2,4,8],difficulty:['all','easy','medium','hard','unknown']};
for(const field of Object.keys(state)){const label=document.createElement('label');label.textContent=names[field];const s=document.createElement('select');s.id=field;for(const v of values[field])s.add(new Option(labels[v]||v,v));s.value=state[field];s.onchange=()=>{state[field]=s.value;render()};label.appendChild(s);document.getElementById('filters').appendChild(label)}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x==null?'—':typeof x==='number'?Number.isInteger(x)?String(x):x.toFixed(5):String(x);
function table(id,rows,columns){document.getElementById(id).innerHTML=rows.length?'<table><thead><tr>'+columns.map(([k,h])=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+columns.map(([k])=>'<td>'+esc(fmt(r[k]))+'</td>').join('')+'</tr>').join('')+'</tbody></table>':'<p>当前分组无结果，或协议不包含该划分。</p>'}
function render(){const matches=r=>Object.keys(state).every(k=>String(r[k])===String(state[k]));const rs=data.means.filter(matches);const ds=data.deltas.filter(matches);document.getElementById('count').textContent=rs.map(r=>r.arm+': '+r.images+' 张 / '+r.fonts+' 字体'+(r.small_font_cell?'（字体数较少）':'')).join(' · ');
table('means',rs,[['arm','模型'],['images','图数'],['fonts','字体数'],['family_s01','Family ↑'],['lpips_alex','LPIPS ↓'],['l1','L1 ↓'],['ssim','SSIM ↑'],['D_region','D_region ↓'],['D_change','D_change ↓'],['D_high','D_high ↓']]);
table('delta',ds,[['comparison','比较'],['metric','指标'],['fonts','字体数'],['delta','差值'],['ci95_low','95%下界'],['ci95_high','95%上界'],['relative_percent','相对变化 %']]);
table('corr',data.correlations.filter(r=>r.script===state.script&&r.difficulty===state.difficulty),[['evaluator','评价器'],['queries_total','查询总数'],['queries_valid','有效查询'],['fonts','字体数'],['rho','Spearman ρ'],['rho_ci95_low','ρ 下界'],['rho_ci95_high','ρ 上界'],['tau','Kendall τ-b'],['tau_ci95_low','τ 下界'],['tau_ci95_high','τ 上界']]);}render();</script></html>'''
    links=' '.join('<a href="'+p.name+'">'+p.name+'</a>' for p in sorted(out.glob('*.csv')))
    if not (out.parent/'index.html').exists():
        page=page.replace('<a href="../index.html">统计总览</a> · <a href="../k_results/index.html">K族图像与逐图分数</a> · ','')
    (out/'index.html').write_text(page.replace('DOWNLOADS',links).replace('PAYLOAD',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metrics',type=Path,nargs='+',required=True)
    ap.add_argument('--e12',type=Path,required=True)
    ap.add_argument('--difficulty',type=Path,required=True)
    ap.add_argument('--legacy-difficulty',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    def save(name,rows):
        f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        f.to_csv(a.out/name,index=False,float_format='%.10g',lineterminator='\n')
        return f
    sources={};allrows=[]
    for p in sorted(p for directory in a.metrics for p in directory.glob('*.json')):
        obj=json.loads(p.read_text())
        if not isinstance(obj.get('rows'),list): continue
        allrows+=obj['rows'];sources[str(p)]=sha(p)
    df=pd.DataFrame(allrows)
    assert not df.duplicated(KEY+['arm']).any()
    # Validate the original paired episode contract, beyond matching row counts.
    bases=df[df.arm=='K0'].set_index(KEY)
    for arm in sorted(set(df.arm)-{'K0'}):
        other=df[df.arm==arm].set_index(KEY)
        assert set(other.index)<=set(bases.index)
        same=bases.loc[other.index]
        for field in ['seed','refs','e12_refs','target','content','script']:
            assert all(x==y for x,y in zip(same[field],other[field])),(arm,field)
    dm=a.difficulty/'difficulty_font_script.csv'
    difficulty=pd.read_csv(dm)
    assert not difficulty.duplicated(['split','font','script']).any()
    df=df.merge(difficulty,on=['split','font','script'],how='left',validate='many_to_one')
    assert df.difficulty.notna().all()
    legacy=json.loads(a.legacy_difficulty.read_text())['font_groups']
    df['legacy_Es_difficulty']=[legacy.get(f+'|'+('latin' if s=='western' else s),{}).get('bucket','unlabelled') for f,s in zip(df.font,df.script)]
    save('per_image_stratified.csv',df[KEY+['arm','script','seed','difficulty','difficulty_score','legacy_Es_difficulty']+METRICS])
    means=[]
    for labels,sub in slices(df,['protocol','split','arm'],['script','k','difficulty']):
        fm=sub.groupby('font')[METRICS].mean()
        means.append(dict(**labels,images=len(sub),fonts=len(fm),font_char_pairs=len(sub[['font','cp']].drop_duplicates()),
                          small_font_cell=len(fm)<5,**fm.mean().to_dict()))
    means=save('metrics_all_strata.csv',means)
    for name,mask in [
        ('metrics_by_shot.csv',(means.script=='all')&(means.difficulty=='all')&(means.k!='all')),
        ('metrics_by_script.csv',(means.script!='all')&(means.difficulty=='all')&(means.k=='all')),
        ('metrics_by_difficulty.csv',(means.script=='all')&(means.difficulty!='all')&(means.k=='all')),
        ('metrics_script_shot_difficulty.csv',(means.script!='all')&(means.difficulty!='all')&(means.k!='all'))]: save(name,means[mask])
    deltas=[]
    for labels,sub in slices(df,['protocol','split'],['script','k','difficulty']):
        base=sub[sub.arm=='K0'].set_index(KEY)
        for arm in sorted(set(sub.arm)-{'K0'}):
            other=sub[sub.arm==arm].set_index(KEY)
            assert set(other.index)<=set(base.index)
            if other.empty: continue
            match=base.loc[other.index]
            dd=(other[METRICS]-match[METRICS]).reset_index().groupby('font')[METRICS].mean()
            low,high=ci(dd.to_numpy());mb=match.reset_index().groupby('font')[METRICS].mean().mean()
            for j,m in enumerate(METRICS):
                delta=float(dd[m].mean())
                deltas.append(dict(**labels,comparison=arm+'-K0',metric=m,images_per_arm=len(other),fonts=len(dd),
                    small_font_cell=len(dd)<5,delta=delta,ci95_low=low[j],ci95_high=high[j],
                    relative_percent=100*delta/mb[m] if mb[m] else np.nan))
    deltas=save('paired_deltas_all_strata.csv',deltas)
    # GT rank validation: one sample per font/codepoint. Generation arm and shot
    # are duplicates for these GT candidates and are explicitly not replicated.
    correlations=[];episodes=[];tau_diffs=[];duplicate_gt_max_difference=0.
    for p in sorted(a.e12.glob('*_scores.json')):
        name=p.stem.removesuffix('_scores');sources[str(p)]=sha(p)
        obj=json.loads(p.read_text());seen={}
        for r in obj['episodes']:
            key=(r['font'],r['cp']);scores=[r['scores'].get(c) for c in 'ANFX']
            if key in seen:
                assert [s is None for s in seen[key]]==[s is None for s in scores]
                gap=max((abs(x-y) for x,y in zip(seen[key],scores) if x is not None),default=0.)
                # Frozen scoring batches may differ by a few FP32 ulps.
                assert gap<=1e-6,(name,key,gap)
                duplicate_gt_max_difference=max(duplicate_gt_max_difference,gap)
                if all(s is not None for s in scores):
                    assert np.array_equal(np.argsort(seen[key]),np.argsort(scores))
                continue
            seen[key]=scores
            valid=all(s is not None for s in scores)
            rho=float(spearmanr([3,2,1,3],scores).statistic) if valid and len(set(scores))>1 else np.nan
            tau=tau_b([3,2,1,3],scores) if valid else np.nan
            if r['tau_GT'] is not None: tau_diffs.append(abs(tau-r['tau_GT']))
            correlations.append(dict(evaluator=name,font=r['font'],cp=r['cp'],script=r['script'],split='test',rho=rho,tau=tau,valid=valid))
        for r in obj['episodes']:
            if name!='R1_cosine': continue
            episodes.append({k:r[k] for k in ['arm','font','cp','script','k','M5','same_char_top2','N_not_last','G_average_rank']} | {'split':'test'})
    assert max(tau_diffs,default=0)<1e-12
    corr=pd.DataFrame(correlations).merge(difficulty,on=['split','font','script'],how='left',validate='many_to_one')
    assert corr.difficulty.notna().all()
    save('e12_correlations_per_query.csv',corr)
    csummary=[]
    for labels,sub in slices(corr,['evaluator'],['script','difficulty']):
        valid=sub.dropna(subset=['rho','tau']);fm=valid.groupby('font')[['rho','tau']].mean()
        low,high=ci(fm.to_numpy())
        csummary.append(dict(**labels,queries_total=len(sub),queries_valid=len(valid),fonts=len(fm),
            rho=float(fm.rho.mean()),rho_ci95_low=low[0],rho_ci95_high=high[0],
            tau=float(fm.tau.mean()),tau_ci95_low=low[1],tau_ci95_high=high[1],small_font_cell=len(fm)<5))
    csummary=save('e12_correlations.csv',csummary)
    ep=pd.DataFrame(episodes).merge(difficulty,on=['split','font','script'],how='left',validate='many_to_one')
    desc=[]
    for labels,sub in slices(ep,['arm'],['script','k','difficulty']):
        valid=sub.dropna(subset=['M5']);fm=valid.groupby('font')[['M5','same_char_top2','N_not_last','G_average_rank']].mean()
        desc.append(dict(**labels,total=len(sub),resolved=len(valid),coverage=len(valid)/len(sub),fonts=len(fm),
                         small_font_cell=len(fm)<5,**fm.mean().to_dict()))
    save('e12_generation_descriptive_strata.csv',desc)
    # Cross-metric agreement is descriptive, separate from evaluator validity.
    # Compare within font/script/shot to avoid a pooled between-font association.
    mc=[]
    for labels,sub in df.groupby(['protocol','split','arm','script','k','font']):
        if len(sub)<8 or sub.family_s01.nunique()<2: continue
        for metric,sign in [('lpips_alex',-1),('l1',-1),('ssim',1),('D_change',-1)]:
            if sub[metric].nunique()<2: continue
            rho=float(spearmanr(sub.family_s01,sign*sub[metric]).statistic)
            mc.append(dict(zip(['protocol','split','arm','script','k','font'],labels)) | dict(metric=metric,higher_is_better_orientation=sign,images=len(sub),rho=rho))
    save('e12_metric_correlation_within_font.csv',mc)
    manifest=dict(aggregation='font macro within each cell; existing split/protocol kept separate',
        ci='paired font bootstrap, 10000 draws, seed 20260917; exploratory pointwise intervals, not multiple-comparison corrected; font<5 flagged; font<2 CI unavailable',
        correlation='per-query Spearman rho and frozen Kendall tau-b on [A,N,F,X] labels [3,2,1,3], then font macro; exact score ties for rho, epsilon1e-6 for frozen tau',
        G_rank_label='unavailable: neither GT nor generation assumed optimal; no invented five-way total order',
        human_correlation='NOT_COLLECTED',GT_correlation_is_proxy=True,
        GT_deduplication='one font/codepoint, not generation model or shot; fixed ref8',
        difficulty_manifest_sha256=sha(a.difficulty/'DIFFICULTY_MANIFEST.json'),legacy_manifest_sha256=sha(a.legacy_difficulty),
        script_sha256=sha(__file__),source_files_sha256=sources,records=len(df),
        matched_pairs=len(df[df.arm=='K1']),strata=len(means),paired_delta_rows=len(deltas),
        frozen_tau_max_difference=max(tau_diffs,default=0),duplicate_GT_max_float_difference=duplicate_gt_max_difference,
        models=sorted(set(df.arm)))
    (a.out/'STATISTICS_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    overview=csummary[(csummary.script=='all')&(csummary.difficulty=='all')]
    lines=['**K族分层统计与 E12-c 排序相关性**','',
        '交互表见 [index.html](index.html)。按 protocol / split / script / shot / GT 重绘难度分别计算，分组内采用字体宏平均。', '',
        'GT 重绘难度使用全部 clean GT 和标准字，先按同一字符的训练字体分布校准，再按字体×script 的中位数分为低、中、高。阈值仅由训练集确定；本定义在 K1 结果后增加，属于探索性诊断。它不等同于人眼复杂度或模型学习难度。', '',
        'K0 与其他模型的目标、参考、噪声、评价参考均逐条核对。配对差按字体重采样10,000次，种子20260917；少于5个字体的分组标注 small_font_cell，少于2个字体不报告区间。多重比较未校正。', '',
        '|评价器|Spearman ρ|ρ 95%区间|Kendall τ-b|有效查询|', '|---|---:|---|---:|---:|']
    for r in overview.to_dict('records'):
        lines.append(f"|{r['evaluator']}|{r['rho']:.5f}|[{r['rho_ci95_low']:.5f}, {r['rho_ci95_high']:.5f}]|{r['tau']:.5f}|{r['queries_valid']}|")
    lines+=['','GT 等级为 A=X>N>F；N/F 是独立代理构造的近/远风格。G 没有预设真值名次，未强行补全五候选总序。GT 查询按字体×字符去重；不同生成器和 shot 不作为独立 GT 样本。','',
        '人评 Spearman/Kendall 仍为 NOT_COLLECTED。当前 GT 排序相关性是代理验证。Family 与负 LPIPS、负 L1、SSIM、负 D_change 的相关性另存为诊断表，也不能替代人评。', '',
        '无并列预测分数时，A/X 的同级标签使完全遵守其余关系的 Kendall τ-b=0.91287、Spearman ρ=0.94868；只有预测分数也精确表达相同并列时才可能达到1。','',
        '当前模型：'+', '.join(sorted(set(df.arm)))+'；总记录 '+str(len(df))+'。K3 若未列出则尚未纳入已完成结果。']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dashboard(a.out,means,deltas,csummary)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_files_sha256']},ensure_ascii=False,indent=2))
    print(csummary[(csummary.script=='all')&(csummary.difficulty=='all')].to_string(index=False))


if __name__=='__main__':main()
