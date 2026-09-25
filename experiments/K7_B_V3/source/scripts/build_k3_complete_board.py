"""Self-contained matched default review with split/font/character/shot filters."""
import hashlib,json,shutil
from pathlib import Path
SOURCE=Path('/root/projects/hrfont/reports/k_default_k1248_20260917')
ROOT=Path('/root/data1/hrfont_k3_complete_review_20260917')
def main():
    ROOT.mkdir(exist_ok=True)
    extra={}
    for arm in ['K0','K1','K3']:
        store=Path('/root/data1/hrfont_k3_paper_metrics_20260917' if arm=='K3' else '/root/data1/hrfont_k_paper_metrics_20260917')
        for sp in ['train','val','test']:
            for r in json.loads((store/('default128_'+arm+'_'+sp+'.json')).read_text())['rows']:extra[sp,arm,r['font'],r['cp'],r['k']]=r
    assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    def asset(path):
        p=Path(path);key=hashlib.sha256(str(p).encode()).hexdigest()[:24]+'.png';dest=assets/key
        if not dest.exists():shutil.copyfile(p,dest)
        return 'assets/'+key
    items=[];summary={}
    for sp in ('train','val','test'):
        tables={arm:json.loads((SOURCE/arm/sp/'metrics.json').read_text()) for arm in ['K0','K1','K3']}
        idx={arm:{(r['font'],r['cp'],r['k']):r for r in d['rows']} for arm,d in tables.items()}
        assert idx['K0'].keys()==idx['K1'].keys()==idx['K3'].keys()
        summary[sp]={arm:d['summary'] for arm,d in tables.items()}
        for f,c in sorted({(f,c) for f,c,k in idx['K0']}):
            base=idx['K0'][f,c,8];row=dict(split=sp,font=f,cp=c,char=chr(int(c[1:],16)),script=base['script'],
                content=asset(base['content']),gt=asset(base['target']),refs=[asset(p) for p in base['refs_paths']],pred={})
            for k in [1,2,4,8]:
                row['pred'][k]={}
                for arm in ['K0','K1','K3']:
                    r=idx[arm][f,c,k];m=extra[sp,arm,f,c,k];row['pred'][k][arm]=dict(src=asset(SOURCE/arm/sp/r['png']),l1=r['l1'],D_change=r['D_change'],Family=m['family_s01'],LPIPS=m['lpips_alex'])
            items.append(row)
    payload=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>HRFont default Train / Val / Test · K0 / K1 / K3</title>
<style>body{font:14px system-ui;margin:22px;color:#18202b;background:#f7f8fa}h1{font-size:24px}button,select,input{margin:4px;padding:7px}table{border-collapse:collapse;background:white}td,th{border:1px solid #ddd;padding:7px;text-align:center}img{width:96px;height:96px;image-rendering:auto}td.refs img{width:38px;height:38px}td.refs{max-width:160px}small{display:block;color:#566}thead{position:sticky;top:0;background:#eff3f8}#bar{display:flex;align-items:center;flex-wrap:wrap}#chars{height:92px;min-width:135px}</style>
<h1>Train / Val / Test · K0 / K1 / K3@10k</h1><p>128候选字符（80西文、32假名、16注音），仅展示合法clean pair。1/2/4/8-shot使用嵌套中文参考与相同噪声。CFG 1，DPM++20。Train为拟合诊断。</p>
<div id="bar"></div><p id="count"></p><button id="prev">上一页</button><button id="next">下一页</button><table><thead id="head"></thead><tbody id="rows"></tbody></table>
<script>const data=PAYLOAD;const state={split:'test',font:'',script:'',chars:[],shots:[1,2,4,8],page:0};const bar=document.getElementById('bar');
for(const field of ['split','font','script']){const s=document.createElement('select');s.innerHTML='<option value="">全部 '+field+'</option>';for(const v of [...new Set(data.map(x=>x[field]))].sort()){const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o)}s.value=state[field];s.onchange=()=>{state[field]=s.value;state.page=0;render()};bar.appendChild(s)}
const chars=document.createElement('select');chars.id='chars';chars.multiple=true;for(const cp of [...new Set(data.map(x=>x.cp))].sort()){const o=document.createElement('option');o.value=cp;o.textContent=String.fromCodePoint(parseInt(cp.slice(1),16))+' '+cp;chars.appendChild(o)}chars.onchange=()=>{state.chars=[...chars.selectedOptions].map(o=>o.value);state.page=0;render()};bar.appendChild(chars);const reset=document.createElement('button');reset.textContent='全部字符';reset.onclick=()=>{for(const o of chars.options)o.selected=false;state.chars=[];state.page=0;render()};bar.appendChild(reset);
for(const k of [1,2,4,8]){const l=document.createElement('label');const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;cb.onchange=()=>{state.shots=cb.checked?[...state.shots,k].sort((a,b)=>a-b):state.shots.filter(x=>x!==k);render()};l.appendChild(cb);l.append(k+'-shot');bar.appendChild(l)}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const img=s=>'<img loading="lazy" src="'+s+'">';
function render(){const rows=data.filter(r=>['split','font','script'].every(f=>!state[f]||r[f]===state[f])&&(!state.chars.length||state.chars.includes(r.cp)));const pages=Math.max(1,Math.ceil(rows.length/40));state.page=Math.min(state.page,pages-1);document.getElementById('count').textContent=rows.length+' 字体字符组合 · 第 '+(state.page+1)+'/'+pages+' 页 · 按住Command/Ctrl可多选字符';document.getElementById('head').innerHTML='<tr><th>样本</th><th>参考</th><th>Content</th><th>GT</th>'+state.shots.map(k=>['K0','K1','K3'].map(a=>'<th>'+a+' '+k+'-shot</th>').join('')).join('')+'</tr>';document.getElementById('rows').innerHTML=rows.slice(state.page*40,(state.page+1)*40).map(r=>'<tr><td>'+esc(r.font)+'<br>'+esc(r.char)+' '+r.cp+'<br>'+r.split+'</td><td class="refs">'+r.refs.map(img).join('')+'</td><td>'+img(r.content)+'</td><td>'+img(r.gt)+'</td>'+state.shots.map(k=>['K0','K1','K3'].map(a=>'<td>'+img(r.pred[k][a].src)+'<small>Family '+r.pred[k][a].Family.toFixed(4)+' · LPIPS '+r.pred[k][a].LPIPS.toFixed(4)+'</small><small>L1 '+r.pred[k][a].l1.toFixed(4)+' · Δ区域 '+r.pred[k][a].D_change.toFixed(4)+'</small></td>').join('')).join('')+'</tr>').join('')};document.getElementById('prev').onclick=()=>{state.page=Math.max(0,state.page-1);render()};document.getElementById('next').onclick=()=>{state.page++;render()};render();</script>'''
    (ROOT/'index.html').write_text(page.replace('PAYLOAD',payload))
    import csv,collections,statistics
    tables=[]
    for sp in ['train','val','test']:
        for arm in ['K0','K1','K3']:
            rr=[r for (s,a,f,c,k),r in extra.items() if s==sp and a==arm]
            for script in ['all','western','kana','bopomofo']:
                for shot in ['all',1,2,4,8]:
                    rs=[r for r in rr if (script=='all' or r['script']==script) and (shot=='all' or r['k']==shot)]
                    fonts=sorted({r['font'] for r in rs})
                    tables.append(dict(split=sp,model=arm,script=script,shot=shot,images=len(rs),fonts=len(fonts),**{key:statistics.mean([statistics.mean([r[key] for r in rs if r['font']==f]) for f in fonts]) for key in ['family_s01','lpips_alex','l1','ssim','D_change']}))
    with (ROOT/'paper_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'DONE.json').write_text(json.dumps(dict(status='completed',pairs=len(items),images=len(items)*12),indent=2)+'\n')
if __name__=='__main__':main()
