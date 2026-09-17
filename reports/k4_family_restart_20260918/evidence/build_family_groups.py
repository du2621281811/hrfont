import collections,csv,hashlib,json,pathlib,re
R=pathlib.Path('repo');A=R/'experiments/K4';O=pathlib.Path('outputs/K4_FAMILY_RESTART_20260918')
rows=list(csv.DictReader((R/'manifests/v2/fonts.tsv').open(),delimiter='\t'))
old=json.loads((A/'aliases.json').read_text());parent={r['stem']:r['stem'] for r in rows};edges=[]
def find(x):
    while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
    return x
def join(a,b,reason,key):
    x,y=find(a),find(b)
    if x!=y:parent[max(x,y)]=min(x,y)
    if a!=b:edges.append(dict(a=a,b=b,reason=reason,key=key))
def union_groups(groups,reason):
    for key,fs in sorted(groups.items()):
        fs=sorted(set(fs))
        for f in fs[1:]:join(fs[0],f,reason,key)
groups=collections.defaultdict(list)
for f,g in old.items():groups[g].append(f)
union_groups(groups,'inherited_alias_and_dataset_group')
groups=collections.defaultdict(list)
for r in rows:
    if r['family'].strip():groups[r['family'].casefold()].append(r['stem'])
union_groups(groups,'dataset_font_family')
suffix=r'[-_](?:EL|UL|EB|UB|DB|B|H|L|M|R|T|Cu|Zhong|Te|Xian|Da|Xi|Zhun|Italic|50[0-9][RMBH]|51[0-9][RMBH])$'
groups=collections.defaultdict(list)
for f in parent:groups[re.sub(suffix,'',f).casefold()].append(f)
union_groups(groups,'explicit_weight_suffix')
names=pathlib.Path('work/dataset_v2_20260917/font_names.json');groups=collections.defaultdict(list)
for r in json.loads(names.read_text()):
    for n in r['names']['16']:groups[n.casefold()].append(r['clean'])
union_groups(groups,'ttf_typographic_family_name16')
mapping={f:find(f) for f in sorted(parent)};members=collections.defaultdict(list)
for f,g in mapping.items():members[g].append(f)
new_merges={g:fs for g,fs in members.items() if len({old[f] for f in fs})>1}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(version='K4-family-exclusion-v1',policy='Exclude target font and all known family/weight/alias siblings before alpha softmax/top-k; strict failure for unknown family or empty eligible bank.',
    family_by_font=mapping,members=dict(members),edges=edges,new_merges=new_merges,
    sources={'inherited_aliases_sha256':sha(A/'aliases.json'),'fonts_tsv_sha256':sha(R/'manifests/v2/fonts.tsv'),'ttf_names_sha256':sha(names)},
    coverage=dict(fonts=len(mapping),families=len(members),multi_member_families=sum(len(fs)>1 for fs in members.values()),fonts_in_multi_member_families=sum(len(fs) for fs in members.values() if len(fs)>1)),
    boundary='Known metadata/explicit-weight/alias families; not a claim that all visually related families are identifiable. Dataset split membership is unchanged.')
(O/'previous_aliases.json').write_text(json.dumps(old,indent=2))
for p in [A/'family_groups.json',O/'family_groups.json']:p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(A/'aliases.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(coverage=manifest['coverage'],new_merges=new_merges),ensure_ascii=False))
