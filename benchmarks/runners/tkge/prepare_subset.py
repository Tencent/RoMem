"""
Create a subsampled ICEWS05-15 dataset for quick local testing.

Usage:  python prepare_subset.py [--max-train 20000]

Outputs data into dataset paths expected by each model:
  - de-simple-master/datasets/icews05-15-sub/
  - ChronoR-main/src_data/ICEWS05-15-SUB/
  - KnowledgeGraphEmbedding-master/data/icews05-15-sub/
"""
import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'dataset' / 'icews05-15'


def read_quads(path):
    """Read tab-separated quadruples: head, rel, tail, timestamp."""
    quads = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                quads.append(tuple(parts[:4]))
    return quads


def filter_by_vocab(quads, entities, relations):
    """Keep only quads whose head, tail, and relation are in the vocab."""
    return [q for q in quads if q[0] in entities and q[2] in entities and q[1] in relations]


def write_quads(quads, path):
    with open(path, 'w') as f:
        for q in quads:
            f.write('\t'.join(q) + '\n')


def write_triples(quads, path):
    """Write static triples (no timestamp), deduplicated."""
    seen = set()
    with open(path, 'w') as f:
        for h, r, t, _ts in quads:
            triple = (h, r, t)
            if triple not in seen:
                seen.add(triple)
                f.write(f'{h}\t{r}\t{t}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-train', type=int, default=20000)
    parser.add_argument('--max-eval', type=int, default=2000,
                        help='Max quads in valid/test sets (0=no cap)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Read full data
    train = read_quads(SRC_DIR / 'train.txt')
    valid = read_quads(SRC_DIR / 'valid.txt')
    test = read_quads(SRC_DIR / 'test.txt')

    print(f'Full data: train={len(train)}, valid={len(valid)}, test={len(test)}')

    # Subsample training data
    if len(train) > args.max_train:
        train = random.sample(train, args.max_train)

    # Collect vocab and timestamps from subsampled train
    entities = set()
    relations = set()
    timestamps = set()
    for h, r, t, ts in train:
        entities.add(h)
        entities.add(t)
        relations.add(r)
        timestamps.add(ts)

    # Filter valid/test to match train vocab AND timestamps
    valid = filter_by_vocab(valid, entities, relations)
    test = filter_by_vocab(test, entities, relations)
    valid = [q for q in valid if q[3] in timestamps]
    test = [q for q in test if q[3] in timestamps]

    # Cap eval set sizes for fast local testing
    if args.max_eval > 0:
        if len(valid) > args.max_eval:
            valid = random.sample(valid, args.max_eval)
        if len(test) > args.max_eval:
            test = random.sample(test, args.max_eval)

    print(f'Subset:    train={len(train)}, valid={len(valid)}, test={len(test)}')
    print(f'           entities={len(entities)}, relations={len(relations)}, '
          f'timestamps={len(timestamps)}')

    # ── Canonical quadruples (used by DE-SimplE and RoMem via symlink) ──
    quad_dir = PROJECT_ROOT / 'dataset' / 'icews05-15-sub'
    quad_dir.mkdir(parents=True, exist_ok=True)
    write_quads(train, quad_dir / 'train.txt')
    write_quads(valid, quad_dir / 'valid.txt')
    write_quads(test, quad_dir / 'test.txt')
    print(f'Quadruples -> {quad_dir}')

    # ── ChronoR ──────────────────────────────────────────────────────────
    chronor_dir = PROJECT_ROOT / 'baselines' / 'tkge' / 'ChronoR-main' / 'src_data' / 'ICEWS05-15-SUB'
    chronor_dir.mkdir(parents=True, exist_ok=True)
    # ChronoR expects files without .txt extension
    write_quads(train, chronor_dir / 'train')
    write_quads(valid, chronor_dir / 'valid')
    write_quads(test, chronor_dir / 'test')
    print(f'ChronoR raw data -> {chronor_dir}')

    # Preprocess for ChronoR (pickle files)
    # process_icews uses relative DATA_PATH='data/', so we must chdir into ChronoR-main
    import sys
    chronor_root = PROJECT_ROOT / 'baselines' / 'tkge' / 'ChronoR-main'
    sys.path.insert(0, str(chronor_root))
    from process_icews import prepare_dataset
    chronor_processed = chronor_root / 'data' / 'ICEWS05-15-SUB'
    if chronor_processed.exists():
        import shutil
        shutil.rmtree(chronor_processed)
    prev_cwd = os.getcwd()
    os.chdir(chronor_root)
    prepare_dataset(str(chronor_dir), 'ICEWS05-15-SUB')
    os.chdir(prev_cwd)
    print(f'ChronoR pickles -> {chronor_processed}')

    # ── RotatE / DistMult (static) ───────────────────────────────────────
    rotate_dir = PROJECT_ROOT / 'baselines' / 'tkge' / 'KnowledgeGraphEmbedding-master' / 'data' / 'icews05-15-sub'
    rotate_dir.mkdir(parents=True, exist_ok=True)

    # Build entity/relation dicts
    ent2id = OrderedDict()
    rel2id = OrderedDict()
    for split_data in [train, valid, test]:
        for h, r, t, _ts in split_data:
            if h not in ent2id:
                ent2id[h] = len(ent2id)
            if t not in ent2id:
                ent2id[t] = len(ent2id)
            if r not in rel2id:
                rel2id[r] = len(rel2id)

    with open(rotate_dir / 'entities.dict', 'w') as f:
        for name, idx in ent2id.items():
            f.write(f'{idx}\t{name}\n')
    with open(rotate_dir / 'relations.dict', 'w') as f:
        for name, idx in rel2id.items():
            f.write(f'{idx}\t{name}\n')

    write_triples(train, rotate_dir / 'train.txt')
    write_triples(valid, rotate_dir / 'valid.txt')
    write_triples(test, rotate_dir / 'test.txt')
    print(f'RotatE data -> {rotate_dir}')

    print('\nDone! Use SUBSET=1 in benchmark.sh to use this data.')


if __name__ == '__main__':
    main()
