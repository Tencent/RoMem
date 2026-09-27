# RoMem: Time is Not a Label

**Continuous Phase Rotation and Geometric Shadowing for Agentic Memory**

[![EMNLP 2026](https://img.shields.io/badge/EMNLP_2026-Main_Conference-8A2D3C.svg)](https://2026.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.11544-b31b1b.svg)](https://arxiv.org/abs/2604.11544)
[![Project Page](https://img.shields.io/badge/Project-Page-2B7A78.svg)](https://tencent.github.io/RoMem/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/Tencent/RoMem?style=flat&logo=github)](https://github.com/Tencent/RoMem/stargazers)

RoMem is a temporal memory framework that internalises time as a geometric physical law rather than a discrete label. It employs continuous phase rotation in complex vector space and a pretrained **Semantic Speed Gate** to separate static truths from dynamic events — enabling agentic memory systems to resolve temporal contradictions without destructive deletion.

[[Paper](https://arxiv.org/abs/2604.11544)] [[Project Page](https://tencent.github.io/RoMem/)] [[Code](https://github.com/Tencent/RoMem)]

---

## News

- **September 2026** — 🎉 RoMem has been selected for an **oral presentation** at EMNLP 2026!
- **August 2026** — 🎉 RoMem was accepted to the **EMNLP 2026 Main Conference**.
- **April 2026** — The [paper](https://arxiv.org/abs/2604.11544) was released on arXiv.

---

## Method Overview

Most agentic memory systems treat knowledge as a static snapshot, leading to catastrophic failures when facts change over time (*"Who is the president?"*). RoMem solves this with three core ideas:

1. **Functional Continuous Time** — Time is a rotation angle `θ(τ) = τ · s · ω · α_r`, not a discrete lookup. Any timestamp (including unseen dates) can be evaluated directly.

2. **Semantic Speed Gate** — An MLP `α_r = σ(MLP(φ(r)))` maps relation text embeddings to a volatility score in (0, 1). Static relations like *"born in"* get α ≈ 0 (no rotation); dynamic relations like *"president of"* get α ≈ 1 (fast rotation). Zero-shot transfer to unseen relations.

3. **Geometric Shadowing** — Obsolete facts are rotated out of phase at query time, causing the temporally correct fact to naturally outrank contradictions — without deletion.

For full details, see our [paper](https://arxiv.org/abs/2604.11544) or visit the [project page](https://tencent.github.io/RoMem/).

---

## Repository Structure

```
romem/                          # Core framework
├── __init__.py                 # Exports: RoMem, RoMemLLM, RoMemReranker, pretrain_gate
├── romem_llm.py                # High-level memory-augmented LLM wrapper
├── romem_reranker.py           # Standalone TKGE temporal reranker
├── pretrain_gate.py            # Convenience function for gate pretraining
├── RoMem.py                    # Full pipeline orchestrator
├── kge/                        # Temporal KGE models, encoder, retriever
│   ├── model/                  # RoMem-ChronoR, RoMem-DistMult architectures
│   ├── encoder.py              # TKGE training loop
│   └── retriever.py            # Scoring and reranking
├── pretraining/                # Semantic speed gate pretraining
├── checkpoints/                # Bundled pretrained gate checkpoints
│   ├── gate_text-embedding-3-small.pt
│   └── gate_bge-m3.pt
├── ingestion/                  # Graph construction + embedding stores
├── prompts/                    # OpenIE and temporal extraction prompts
└── configs/                    # Dataset-specific configurations

benchmarks/                     # Evaluation framework
├── runners/                    # Per-benchmark × per-framework runners
├── evaluators/                 # Metric computation per benchmark
├── loaders/                    # Dataset loaders
└── reports/                    # Experiment results (JSONL)

baselines/                      # Baseline implementations
├── HippoRAG/                   # Knowledge graph RAG with PPR
├── mem0/                       # Vector memory with FAISS
├── graphiti/                   # Temporal KG with Neo4j
└── tkge/                       # TKGE baselines (ChronoR, DE-SimplE, RotatE)

eval/                           # CLI entry points for benchmarks
scripts/                        # Reproducible run scripts
dataset/                        # Raw datasets
```

---

## Quick Start

### Installation

```bash
# Python 3.10+ required
pip install -r requirements-romem.txt
```

### Option 1: Memory-Augmented LLM (3 lines)

```python
from romem import RoMemLLM

llm = RoMemLLM(
    llm="gpt-4o-mini",                      # or a local model via vLLM
    embedding="text-embedding-3-small",       # bundled gate checkpoint
    save_dir="./my_memory",
)

llm.add("Obama was president from 2009 to 2017.", timestamp="2017-01-20")
llm.add("Biden became president in January 2021.", timestamp="2021-01-20")

answer = llm.ask("Who was president in 2019?")    # → Obama
answer = llm.ask("Who is the current president?") # → Biden
```

### Option 2: Temporal Reranker (Plug into Your Pipeline)

Use the TKGE module as a drop-in temporal reranker for any retrieval system:

```python
from romem import RoMemReranker

reranker = RoMemReranker(embedding="text-embedding-3-small")

# Train on temporal facts from ICEWS05-15 — competing facts in the same
# (head, relation) slot teach the model temporal discrimination.
reranker.fit([
    ("Barack Obama", "Consult", "Tony Blair",    "2007-12-21"),
    ("Barack Obama", "Consult", "Mahmoud Abbas",  "2008-07-15"),
    ("Barack Obama", "Consult", "Xi Jinping",     "2013-06-08"),
    ("Barack Obama", "Consult", "Xi Jinping",     "2014-11-11"),
    ("Barack Obama", "Consult", "Xi Jinping",     "2015-04-19"),
])

# Score candidates at a query time
scores = reranker.score(
    candidates=[
        ("Barack Obama", "Consult", "Tony Blair"),
        ("Barack Obama", "Consult", "Xi Jinping"),
    ],
    query_time="2008-01-01",
)
# → Tony Blair scores higher (Obama consulted Blair in 2007-2008)

scores = reranker.score(
    candidates=[
        ("Barack Obama", "Consult", "Tony Blair"),
        ("Barack Obama", "Consult", "Xi Jinping"),
    ],
    query_time="2014-06-01",
)
# → Xi Jinping scores higher (Obama consulted Xi in 2013-2015)

# Fuse with your own semantic scores: S_final = S_sem * (1 + alpha * S_kge)
results = reranker.rerank(
    candidates=my_candidates,
    semantic_scores=my_scores,
    query_time="2019-06-01",
    alpha=0.3,
)

# Inspect gate values (pretrained on ICEWS05-15)
reranker.get_alpha("Consult")        # → ~0.87 (highly dynamic)
reranker.get_alpha("Make a visit")   # → ~0.55 (moderately dynamic)
reranker.get_alpha("Make statement") # → ~0.39 (more static)

# Save / load
reranker.save("./my_reranker")
reranker = RoMemReranker.load("./my_reranker")
```

### Bundled Gate Checkpoints

The semantic speed gate is pretrained on ICEWS05-15 (251 relations, ~460K temporal facts). Two checkpoints are bundled:

| Embedding Model | Checkpoint | Auto-loaded |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | `gate_text-embedding-3-small.pt` | Yes |
| `BAAI/bge-m3` (open-source) | `gate_bge-m3.pt` | Yes |

For other embedding models, pretrain your own gate:

```python
from romem import pretrain_gate

pretrain_gate(
    embedding="nomic-ai/nomic-embed-text-v1.5",
    icews_dir="dataset/icews05-15",    # raw ICEWS05-15 quadruples
    output="./my_gate.pt",
    epochs=100,
)
```

The function handles everything end-to-end: mines temporal transitions from ICEWS05-15 → computes relation embeddings with your model → trains the gate MLP → saves checkpoint.

---

## Reproducing Paper Results

### Environment Setup

```bash
pip install -r requirements-romem.txt
set -a; source .env; set +a        # load API keys
export PYTHONPATH=.
```

### RQ1: TKGE Verification (ICEWS05-15)

Verify that functional temporal modelling preserves TKGE accuracy:

```bash
# Full benchmark (all baselines + RoMem variants)
bash scripts/tkge_implementation/benchmark.sh

# Results summary
bash scripts/tkge_implementation/benchmark.sh summary
```

### RQ2: Agentic Memory Benchmarks

We evaluate across a three-tier temporal spectrum under two implementation configs (OpenAI API and open-source).

**MultiTQ** (heavy temporal reasoning, 500 questions over 11K facts):
```bash
bash scripts/romem_eval/run-multitq-openai.sh     # GPT-5-mini + text-embedding-3-small
bash scripts/romem_eval/run-multitq-server.sh      # LLaMA-3.1-70B + BGE-M3
```

**LoCoMo** (hybrid reasoning, 1986 questions):
```bash
bash scripts/romem_eval/run-locomo-openai.sh
bash scripts/romem_eval/run-locomo-server.sh
```

**DMR-MSC** (static memory preservation, 500 dialogues):
```bash
bash scripts/romem_eval/run-dmr-msc-openai.sh
bash scripts/romem_eval/run-dmr-msc-server.sh
```

### RQ3: Domain Generalisation (FinTMMBench)

```bash
bash scripts/romem_eval/run-fintmmbench-openai.sh
bash scripts/romem_eval/run-fintmmbench-server.sh
```

### Running Baselines

Each baseline has its own eval scripts:

```bash
bash scripts/hipporag_eval/run-locomo-hipporag.sh
bash scripts/mem0_eval/run-locomo-mem0.sh
bash scripts/licomem_eval/run-locomo-licomem.sh
bash scripts/amem_eval/run-locomo-amem.sh
```

### Gate Pretraining (from scratch)

```bash
# Step 1: Mine transition observations from ICEWS05-15
python -m romem.pretraining.build_gate_pretrain_data \
  --out-dir outputs/pretrain_gate_data \
  --icews-dir dataset/icews05-15

# Step 2: Train the gate MLP
python -m romem.pretraining.pretrain_alpha_r \
  --pretrain-data-dir outputs/pretrain_gate_data \
  --embedding-model text-embedding-3-small \
  --epochs 100
```

Or use the convenience function:

```python
from romem import pretrain_gate
pretrain_gate(embedding="text-embedding-3-small", icews_dir="dataset/icews05-15")
```

---

## Configuration

Key configuration options (set via config files in `romem/configs/` or environment variables):

| Setting | Description | Default |
|---|---|---|
| `temporal_awareness` | Master switch for temporal pipeline | `"romem"` |
| `enable_tkge_tunnel` | Enable TKGE reranking | `True` |
| `tkge_temporal_mode` | Temporal mode (`"romem"` or `"none"`) | `"romem"` |
| `tkge_weight` | Temporal reranking weight (α_g) | `0.3` |
| `tkge_time_contrastive_weight` | Time contrastive loss weight (λ_t) | `0.5` |

Full hyperparameter details are in the paper appendix.

[//]: # (---)

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{li2026romem,)

[//]: # (  title={Time is Not a Label: Continuous Phase Rotation and Geometric Shadowing for Agentic Memory},)

[//]: # (  author={Li, Weixian Waylon and Yang, Xianan Jim},)

[//]: # (  year={2026})

[//]: # (})

[//]: # (```)
