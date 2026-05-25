"""
Prepare ICEWS05-15 data for RotatE (static KG format).
Strips timestamps, creates entities.dict and relations.dict.
"""
import os
from collections import OrderedDict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..', '..')
SRC_DIR = os.path.join(_PROJECT_ROOT, 'dataset', 'icews05-15')
DST_DIR = os.path.join(_PROJECT_ROOT, 'baselines', 'tkge', 'KnowledgeGraphEmbedding-master', 'data', 'icews05-15')

os.makedirs(DST_DIR, exist_ok=True)

entities = OrderedDict()
relations = OrderedDict()

# First pass: collect all entities and relations
for split in ['train', 'valid', 'test']:
    with open(os.path.join(SRC_DIR, f'{split}.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            h, r, t = parts[0], parts[1], parts[2]
            if h not in entities:
                entities[h] = len(entities)
            if t not in entities:
                entities[t] = len(entities)
            if r not in relations:
                relations[r] = len(relations)

# Write dictionaries
with open(os.path.join(DST_DIR, 'entities.dict'), 'w') as f:
    for name, idx in entities.items():
        f.write(f'{idx}\t{name}\n')

with open(os.path.join(DST_DIR, 'relations.dict'), 'w') as f:
    for name, idx in relations.items():
        f.write(f'{idx}\t{name}\n')

# Write static triples (strip timestamps, deduplicate)
for split in ['train', 'valid', 'test']:
    seen = set()
    with open(os.path.join(SRC_DIR, f'{split}.txt')) as fin, \
         open(os.path.join(DST_DIR, f'{split}.txt'), 'w') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            h, r, t = parts[0], parts[1], parts[2]
            triple = (h, r, t)
            if triple not in seen:
                seen.add(triple)
                fout.write(f'{h}\t{r}\t{t}\n')

print(f'Entities: {len(entities)}, Relations: {len(relations)}')
print(f'Data written to {DST_DIR}')
