"""Build explicit weight-variant groups independently of split/alias lineage."""
import collections, csv, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Italic, width, encoding editions and alias relationships are NOT weight suffixes.
WEIGHT_SUFFIX=re.compile(r'[-_](?:EL|UL|EB|UB|DB|B|H|L|M|R|T|Cu|Zhong|Te|Xian|Da|Xi|Zhun|50[0-9][RMBH]|51[0-9][RMBH])$')
def build():
    source=ROOT/'manifests/v2/fonts.tsv'
    rows=list(csv.DictReader(source.open(),delimiter='\t'))
    groups=collections.defaultdict(list)
    for r in rows:groups[WEIGHT_SUFFIX.sub('',r['stem'])].append(r['stem'])
    members={min(fs):sorted(fs) for fs in groups.values()}
    mapping={f:g for g,fs in members.items() for f in fs}
    return dict(version='K4-weight-only-v1',policy='Exclude self and explicit same-design weight variants BEFORE alpha softmax/top-k. No alias, edition, italic or visual-similarity union.',family_by_font=dict(sorted(mapping.items())),members=dict(sorted(members.items())),sources={'fonts_tsv_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},weight_suffix_pattern=WEIGHT_SUFFIX.pattern,coverage=dict(fonts=len(mapping),groups=len(members),multi_member_groups=sum(len(fs)>1 for fs in members.values()),fonts_in_multi_member_groups=sum(len(fs) for fs in members.values() if len(fs)>1)),boundary='Operational explicit weight grouping for the present inventory; not a guarantee against all style similarity or historical parent exposure. Dataset and auxiliary lineage groups remain separate.')
if __name__=='__main__':
    result=build();(ROOT/'experiments/K4/weight_groups.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['coverage'])
