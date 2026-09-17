#!/usr/bin/env python3
"""Portable full-glyph review for frozen v0913 + independently split v0917."""
from __future__ import annotations
import argparse
import csv
import html
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def cp(c):return f'u{ord(c):04X}'


def sheet(root, out, font, split, chars, kind, valid, missing, rejected, label):
    columns=16;cellw=100;cellh=114;top=42
    im=Image.new('RGB',(columns*cellw,top+((len(chars)+columns-1)//columns)*cellh),'#e2e8f0')
    d=ImageDraw.Draw(im);ft=ImageFont.load_default()
    d.text((12,12),label,fill='#0f172a',font=ft)
    for i,c in enumerate(chars):
        code=cp(c);x=(i%columns)*cellw;y=top+(i//columns)*cellh
        p=root/'v2'/split/kind/font/f'{font}+{code}.png'
        if code in valid:
            assert p.is_file(),p
            with Image.open(p) as g:im.paste(g.convert('RGB'),(x+2,y+1))
            tag=code;fill='#0f172a'
        else:
            color='#fed7aa' if code in missing else '#e9d5ff' if code in rejected else '#cbd5e1'
            d.rectangle((x+2,y+1,x+97,y+96),fill=color)
            msg='MISSING' if code in missing else 'REJECTED' if code in rejected else 'NOT ELIGIBLE'
            d.text((x+7,y+42),msg,fill='#475569',font=ft);tag=code;fill='#475569'
        d.text((x+18,y+98),tag,fill=fill,font=ft)
    im.save(out,optimize=True)


TEMPLATE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DATASET 数据集 Review</title><style>
*{box-sizing:border-box}body{font:15px/1.65 system-ui,sans-serif;margin:0;background:#f1f5f9;color:#172033}header{background:#102a43;color:#fff;padding:25px 5vw}header h1{margin:0}header a{color:#a5f3fc}main{max-width:1600px;margin:auto;padding:24px}p{margin:8px 0}select,input,button{padding:9px;border:1px solid #cbd5e1;border-radius:6px;background:white;font:inherit}nav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{background:white;padding:12px 20px;border-radius:8px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;text-align:left;border-bottom:1px solid #e2e8f0}tr[data-font]{cursor:pointer}tr[data-font]:hover,tr.selected{background:#e0f2fe}#list{max-height:410px;overflow:auto;border:1px solid #cbd5e1}.panel{padding:18px;background:white;margin-top:18px;border-radius:9px}.panel img{width:100%;height:auto;display:block}code{word-break:break-all}.pill{background:#e0f2fe;padding:3px 7px;border-radius:5px}.notice{padding:12px;border-left:4px solid #d97706;background:#fffbeb}a{color:#0369a1}#styleBox{display:none}button{cursor:pointer}h2{margin:8px 0}footer{margin:25px 0;color:#64748b}.muted{color:#64748b}
</style><header><h1>DATASET · 划分与完整字形审阅</h1><p>0913 固定不变；0917 先独立划分，再按同名 split 合并。全部图片来自实际数据集。</p><a href="index.html">合并 v2</a> · <a href="v0917.html">独立 0917</a> · <a href="REPORT.md">构建报告</a> · <a href="VERIFICATION.json">验证结果</a> · <a href="font_inventory.tsv">逐字体清单</a></header>
<main><div class="metrics" id="metrics"></div><p class="notice">这是待你审阅的数据版本，尚未用于新训练。灰色格=未授权语种/旧版排除；橙色格=已知缺字；紫色格=自动检查拒绝。这些格子都不进入 target GT。点击字形图可查看原尺寸。</p>
<nav><select id="split"><option value="">全部 split</option><option>train</option><option>val</option><option>test</option></select><select id="source"><option value="">全部来源</option><option value="0917">0917 新增</option><option value="0913">0913 原有</option></select><select id="script"><option value="">全部语种</option><option value="latin">西文</option><option value="kana">假名</option><option value="bopomofo">注音</option></select><input id="q" placeholder="字体名 / stem / family" size="35"><label><input id="gaps" type="checkbox">仅缺字或异常</label><span id="count"></span></nav>
<div id="list"><table><thead><tr><th>字体</th><th>来源</th><th>split</th><th>有效 GT</th><th>中文参考</th><th>备注</th></tr></thead><tbody id="fonts"></tbody></table></div>
<section id="detail" class="panel"><h2 id="title"></h2><p id="meta"></p><p id="policy"></p><p>全量目标字符：数字10＋大小写52＋扩展拉丁27＋平假名83＋片假名86＋注音37；每格保留 96×96 原生图像。</p><a id="targetLink" target="_blank"><img id="target" alt="该字体所有目标字符的实际GT及逐字排除状态"></a><p><button id="styleToggle">展开全部 338 个中文参考</button></p><div id="styleBox"><a id="styleLink" target="_blank"><img id="style" alt="全部中文参考图像"></a></div></section>
<footer>冻结种子 3407。同系列分组包含官方 family、TTF typographic family、明确字重后缀、ASCII62＋中文8完全相同图像。历史0913已有跨集合系列保留并在报告中披露；这不等于证明全数据无历史泄漏。</footer></main>
<script>const DATA=PAYLOAD;const view='VIEW';const E=id=>document.getElementById(id);const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let selected=null;
const pool=DATA.fonts.filter(x=>view==='v2'||x.source==='0917');E('metrics').innerHTML=['train','val','test'].map(s=>{const fs=pool.filter(x=>x.split===s);return `<div class="metric"><b>${s}</b>　${fs.length} 字体（${fs.filter(x=>x.target_n>0).length} 有效）<br>${fs.reduce((a,x)=>a+x.target_n,0).toLocaleString()} target GT</div>`}).join('');
function selectFont(f){const x=pool.find(x=>x.stem===f);if(!x)return;selected=f;E('title').textContent=x.display_name+' · '+x.stem;E('meta').textContent=`${x.source} / ${x.split} / ${x.target_n} GT / ${x.style_n} 中文参考 / family=${x.family}`;E('policy').textContent=`${x.note} ${x.missing_text?'缺字：'+x.missing_text:''}`;E('target').src=x.target_sheet;E('targetLink').href=x.target_sheet;E('style').src=x.style_sheet;E('styleLink').href=x.style_sheet;E('styleBox').style.display='none';E('styleToggle').textContent='展开全部 338 个中文参考';document.querySelectorAll('tr[data-font]').forEach(r=>r.classList.toggle('selected',r.dataset.font===f));history.replaceState(null,'','#'+encodeURIComponent(f));}
function render(){const q=E('q').value.toLowerCase();const fs=pool.filter(x=>(!E('split').value||x.split===E('split').value)&&(!E('source').value||x.source===E('source').value)&&(!E('script').value||x.groups.includes(E('script').value))&&(!E('gaps').checked||x.has_issue)&&(x.stem+' '+x.display_name+' '+x.family).toLowerCase().includes(q));E('count').textContent=fs.length+' 字体';E('fonts').innerHTML=fs.map(x=>`<tr data-font="${esc(x.stem)}"><td>${esc(x.display_name)}<br><small>${esc(x.stem)}</small></td><td>${x.source}</td><td>${x.split}</td><td>${x.target_n}</td><td>${x.style_n}</td><td>${esc(x.note)}</td></tr>`).join('');document.querySelectorAll('tr[data-font]').forEach(r=>r.onclick=()=>selectFont(r.dataset.font));if(fs.length&&!fs.some(x=>x.stem===selected))selectFont(fs[0].stem);else if(fs.length)selectFont(selected);else E('detail').style.display='none';if(fs.length)E('detail').style.display='block';}
['split','source','script','q','gaps'].forEach(id=>E(id).addEventListener(id==='q'?'input':'change',render));E('styleToggle').onclick=()=>{const open=E('styleBox').style.display==='block';E('styleBox').style.display=open?'none':'block';E('styleToggle').textContent=open?'展开全部 338 个中文参考':'收起中文参考';};const hash=decodeURIComponent(location.hash.slice(1));if(pool.some(x=>x.stem===hash))selected=hash;render();
</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);a=p.parse_args()
    root=a.root;out=root/'review';(out/'sheets').mkdir(parents=True,exist_ok=True)
    charset=read(a.contracts/'manifests/charset_cn2west_v2_planned.json')
    fs=rows(root/'manifests/v2/fonts.tsv');byfont={f['stem']:[] for f in fs}
    for s in ('train','val','test'):
        for r in rows(root/'manifests/v2'/f'pairs_{s}.tsv'):byfont[r['font']].append(r)
    summary=[]
    def process(f):
        stem=f['stem'];split=f['split'];new=read(root/'rendered'/stem/'record.json') if f['source']=='0917' else None
        target_valid={x['cp'] for x in byfont[stem]}
        style_valid={p.stem.split('+')[-1] for p in (root/'v2'/split/'StyleImage'/stem).glob('*.png')}
        missing={cp(c) for chars in (new['missing_chars'].values() if new else []) for c in chars}
        rejected={x['cp'] for x in new['render_rejections'] if x['kind']=='target'} if new else set()
        tag='sheets/'+stem
        sheet(root,out/(tag+'_target.png'),stem,split,charset['target_string'],'TargetImage',target_valid,missing,rejected,f'{stem} | {f["source"]} | {split} | TARGET: {len(target_valid)} / 295')
        style_missing={cp(c) for c in new['style_missing_cmap']} if new else set()
        style_rejected={x['cp'] for x in new['render_rejections'] if x['kind']=='style'} if new else set()
        sheet(root,out/(tag+'_style.png'),stem,split,charset['style_han_338'],'StyleImage',style_valid,style_missing,style_rejected,f'{stem} | {f["source"]} | {split} | CHINESE REFERENCE: {len(style_valid)} / 338')
        note='0913 原划分、原GT保留' if not new else '按人工勾选语种生成GT'
        if missing:note+='；已知缺字已跳过'
        if rejected:note+='；自动拒绝字符已跳过'
        if len(style_valid)<338:note+='；中文参考有缺字，须用实际参考池'
        mean_bbox=sum(x['bbox_ratio'] for x in new['images'])/len(new['images']) if new else None
        if mean_bbox is not None and mean_bbox<.20:note+='；字形偏小，请重点审阅'
        if not target_valid:note='0913 原排除字体，保留归属，不参与训练/评估'
        r=dict(**f,target_n=len(target_valid),style_n=len(style_valid),groups=sorted({x['script_group'] for x in byfont[stem]}),
               target_sheet=tag+'_target.png',style_sheet=tag+'_style.png',has_issue=bool(missing or rejected or len(style_valid)<338 or not target_valid or (mean_bbox is not None and mean_bbox<.20)),
               render_font_size=new['size'] if new else None,mean_bbox_ratio=mean_bbox,
               missing_text=''.join(new['missing_chars'].values()) if new else '',note=note)
        return r
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(process,fs),1):
            summary.append(r)
            if i%40==0 or i==len(fs):print(f'review sheets {i}/{len(fs)}',flush=True)
    payload=json.dumps(dict(fonts=summary),ensure_ascii=False).replace('</','<\\/')
    for name,view,title in [('index.html','v2','v2'),('v0917.html','v0917','0917')]:
        (out/name).write_text(TEMPLATE.replace('PAYLOAD',payload).replace('VIEW',view).replace('DATASET',title))
    (out/'font_inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (out/'font_inventory.tsv').open('w',newline='') as f:
        fields=['stem','display_name','source','split','family','target_n','style_n','missing_text','note'];w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(summary)
    for n in ['VERIFICATION.json','ASSEMBLED.json','split_audit.json','documented_missing.tsv','new_render_rejections.tsv']:
        if (root/n).exists():shutil.copy2(root/n,out/n)
    shutil.copytree(root/'manifests',out/'manifests',dirs_exist_ok=True)
    print('review ready',out,flush=True)


if __name__=='__main__':main()
