#!/usr/bin/env python3
"""
One-time script: compute OpenAI text-embedding-3-large embeddings
for all ICEWS05-15 relation names and save them to disk.

Usage:
    # Load API key and base URL from .env first
    export $(grep -E '^OPENAI_API_KEY|^OPENAI_BASE_URL' ../.env | xargs)
    python cache_relation_embeddings.py --dataset-dir ../dataset/icews05-15

Output: benchmarks/runners/tkge/data/icews_relation_embeddings.pt
    A dict with keys:
        "relation2id": OrderedDict  (relation_name -> int)
        "embeddings":  Tensor [num_rel, 1536]
        "model":       str  (embedding model name)
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent


def load_relations(dataset_dir):
    """Collect unique relations from all splits, ordered by first appearance in train."""
    rel2id = OrderedDict()
    for split in ["train.txt", "valid.txt", "test.txt"]:
        with open(dataset_dir / split) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    rel = parts[1]
                    if rel not in rel2id:
                        rel2id[rel] = len(rel2id)
    return rel2id


def relation_text(rel: str) -> str:
    """Match TKGEEncoder._relation_text() preprocessing."""
    rel = str(rel).strip()
    if rel.endswith("_inv"):
        rel = rel[:-4]
    return rel.replace("_", " ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(SCRIPT_DIR.parent.parent.parent / "dataset" / "icews05-15"))
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--dimensions", type=int, default=1536,
                        help="Output dimensionality (must match pretrained gate checkpoint)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable first.")

    dataset_dir = Path(args.dataset_dir)
    rel2id = load_relations(dataset_dir)
    print(f"Found {len(rel2id)} unique relations")

    # Prepare texts
    texts = [relation_text(r) for r in rel2id.keys()]
    print(f"Sample texts: {texts[:5]}")

    # Call OpenAI API (supports custom base URL via OPENAI_BASE_URL)
    from openai import OpenAI
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url)
    if base_url:
        print(f"Using custom base URL: {base_url}")

    print(f"Embedding {len(texts)} relations with {args.model} (dim={args.dimensions})...")
    response = client.embeddings.create(
        input=texts, model=args.model, dimensions=args.dimensions)
    embeddings = [item.embedding for item in response.data]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    print(f"Raw embedding shape: {emb_tensor.shape}")

    # Truncate to target dimensions if API returned more (Matryoshka truncation)
    if emb_tensor.shape[1] > args.dimensions:
        print(f"Truncating {emb_tensor.shape[1]} -> {args.dimensions} dims (Matryoshka)")
        emb_tensor = emb_tensor[:, :args.dimensions]
        # Re-normalize after truncation
        emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
    print(f"Final embedding shape: {emb_tensor.shape}")

    # Save
    out_dir = SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "icews_relation_embeddings.pt"
    torch.save({
        "relation2id": rel2id,
        "embeddings": emb_tensor,
        "model": args.model,
        "dimensions": args.dimensions,
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
