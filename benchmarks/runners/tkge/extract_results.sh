#!/usr/bin/env bash
# ===========================================================================
#  Unified TKG Benchmark on ICEWS05-15
#  Runs: DE-SimplE, ChronoR, RotatE, DistMult + RoMem variants
#
#  Usage:  ./benchmark.sh [model ...]
#          ./benchmark.sh                  # run all models
#          ./benchmark.sh chronor rotate   # run only selected models
#          ./benchmark.sh summary          # print results table from logs
#
#  Environment variables:
#    CUDA_DEVICE=0       GPU id (default: 0, ignored on CPU)
#    QUICK=1             Reduced epochs/steps for quick local test
#    SUBSET=1            Use 20K-triple subset (run prepare_subset.py first)
# ===========================================================================
set -euo pipefail

# Force line-buffered Python output so logs appear in real time through `| tee`
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${SCRIPT_DIR}/.venv"
RESULTS_DIR="${SCRIPT_DIR}/results"
QUICK="${QUICK:-0}"
SUBSET="${SUBSET:-0}"

# ── dataset names (full vs subset) ──────────────────────────────────────────
if [ "${SUBSET}" = "1" ]; then
    DE_DATASET="icews05-15-sub"
    CHRONOR_DATASET="ICEWS05-15-SUB"
    ROTATE_DATA="data/icews05-15-sub"
    ROMEM_DATA="${SCRIPT_DIR}/de-simple-master/datasets/icews05-15-sub"
else
    DE_DATASET="icews05-15"
    CHRONOR_DATASET="ICEWS05-15"
    ROTATE_DATA="data/icews05-15"
    ROMEM_DATA="${SCRIPT_DIR}/../dataset/icews05-15"
fi

# ── device detection ─────────────────────────────────────────────────────────
detect_device() {
    "${VENV}/bin/python" -c "
import torch
if torch.cuda.is_available():
    print('cuda')
else:
    print('cpu')
"
}

USE_DEVICE="$(detect_device)"

if [ "${USE_DEVICE}" = "cuda" ]; then
    DEVICE="${CUDA_DEVICE:-0}"
    export CUDA_VISIBLE_DEVICES="${DEVICE}"
    export CUDA_DEVICE=0
    ROTATE_CUDA_FLAG="--cuda"
else
    ROTATE_CUDA_FLAG=""
fi

mkdir -p "${RESULTS_DIR}"

# ── training sizes ───────────────────────────────────────────────────────────
# Full run (server/GPU) vs quick run (local/CPU sanity check)
if [ "${QUICK}" = "1" ]; then
    DE_EPOCHS=60
    DE_SAVE_EACH=10
    DE_NEG_RATIO=20
    CHRONOR_EPOCHS=60
    CHRONOR_VALID_FREQ=5
    ROTATE_STEPS=2000
    ROTATE_VALID_STEPS=1000
    ROTATE_SAVE_STEPS=1000
    ROMEM_EPOCHS=60
    ROMEM_VALID_FREQ=10
else
    DE_EPOCHS=500
    DE_SAVE_EACH=20
    DE_NEG_RATIO=500
    CHRONOR_EPOCHS=200
    CHRONOR_VALID_FREQ=5
    ROTATE_STEPS=80000
    ROTATE_VALID_STEPS=10000
    ROTATE_SAVE_STEPS=10000
    ROMEM_EPOCHS=500
    ROMEM_VALID_FREQ=20
fi

# ── helpers ──────────────────────────────────────────────────────────────────
activate_venv() { source "${VENV}/bin/activate"; }

log() { echo -e "\n========== $1 ==========\n"; }

# ── model runners ────────────────────────────────────────────────────────────
run_de_simple() {
    log "DE-SimplE on ${DE_DATASET} (${USE_DEVICE})"
    cd "${SCRIPT_DIR}/de-simple-master"
    activate_venv
    python main.py \
        -dataset "${DE_DATASET}" \
        -model DE_SimplE \
        -ne ${DE_EPOCHS} -bsize 512 -lr 0.001 \
        -emb_dim 100 -neg_ratio ${DE_NEG_RATIO} \
        -dropout 0.4 -se_prop 0.68 \
        -save_each ${DE_SAVE_EACH} \
        2>&1 | tee "${RESULTS_DIR}/de_simple.log"
    cd "${SCRIPT_DIR}"
}

run_chronor() {
    log "ChronoR on ${CHRONOR_DATASET} (${USE_DEVICE})"
    cd "${SCRIPT_DIR}/ChronoR-main"
    activate_venv

    # Preprocess if needed
    if [ ! -f "data/${CHRONOR_DATASET}/train.pickle" ]; then
        python -c "
from process_icews import prepare_dataset
import os
prepare_dataset(os.path.join('src_data', '${CHRONOR_DATASET}'), '${CHRONOR_DATASET}')
"
    fi

    python learner.py \
        --dataset "${CHRONOR_DATASET}" \
        --model ChronoR \
        --max_epochs ${CHRONOR_EPOCHS} \
        --valid_freq ${CHRONOR_VALID_FREQ} \
        --rank 800 --k 3 --ratio 0.1 \
        --batch_size 1000 \
        --learning_rate 0.1 \
        --emb_reg 0.01 --time_reg 0.01 \
        2>&1 | tee "${RESULTS_DIR}/chronor.log"
    cd "${SCRIPT_DIR}"
}

run_rotate() {
    log "RotatE (static) on ${ROTATE_DATA} (${USE_DEVICE})"
    cd "${SCRIPT_DIR}/KnowledgeGraphEmbedding-master"
    activate_venv

    # Prepare data if needed (only for full dataset; subset is created by prepare_subset.py)
    if [ "${SUBSET}" != "1" ] && [ ! -f "data/icews05-15/entities.dict" ]; then
        cd "${SCRIPT_DIR}"
        python prepare_rotate_data.py
        cd "${SCRIPT_DIR}/KnowledgeGraphEmbedding-master"
    fi

    python codes/run.py \
        --do_train --do_valid --do_test \
        ${ROTATE_CUDA_FLAG} \
        --data_path "${ROTATE_DATA}" \
        --model RotatE \
        -n 256 -b 1024 -d 500 \
        -g 24.0 -a 1.0 -adv \
        -lr 0.0001 \
        --max_steps ${ROTATE_STEPS} \
        --valid_steps ${ROTATE_VALID_STEPS} \
        --save_checkpoint_steps ${ROTATE_SAVE_STEPS} \
        --test_batch_size 16 \
        -de \
        -save "${RESULTS_DIR}/rotate_model" \
        2>&1 | tee "${RESULTS_DIR}/rotate.log"
    cd "${SCRIPT_DIR}"
}

run_distmult() {
    log "DistMult (static) on ${ROTATE_DATA} (${USE_DEVICE})"
    cd "${SCRIPT_DIR}/KnowledgeGraphEmbedding-master"
    activate_venv

    python codes/run.py \
        --do_train --do_valid --do_test \
        ${ROTATE_CUDA_FLAG} \
        --data_path "${ROTATE_DATA}" \
        --model DistMult \
        -n 256 -b 1024 -d 500 \
        -g 200.0 -a 1.0 -adv \
        -lr 0.001 -r 0.00001 \
        --max_steps ${ROTATE_STEPS} \
        --valid_steps ${ROTATE_VALID_STEPS} \
        --save_checkpoint_steps ${ROTATE_SAVE_STEPS} \
        --test_batch_size 16 \
        -save "${RESULTS_DIR}/distmult_model" \
        2>&1 | tee "${RESULTS_DIR}/distmult.log"
    cd "${SCRIPT_DIR}"
}

run_romem() {
    log "RoMem (temporal + pretrained gate) on ${ROMEM_DATA} (${USE_DEVICE})"
    activate_venv
    "${VENV}/bin/python" "${SCRIPT_DIR}/romem_tkg_benchmark.py" \
        --dataset-dir "${ROMEM_DATA}" \
        --model romem \
        --embedding-dim 64 \
        --epochs ${ROMEM_EPOCHS} \
        --batch-size 512 \
        --lr 0.001 \
        --num-negatives 128 \
        --num-conflict-negatives 1 \
        --margin 0.5 \
        --validate-every ${ROMEM_VALID_FREQ} \
        --time-contrastive-weight 0.5 \
        2>&1 | tee "${RESULTS_DIR}/romem.log"
}

run_romem_no_tc() {
    log "RoMem-NoTC (no time contrastive) on ${ROMEM_DATA} (${USE_DEVICE})"
    activate_venv
    "${VENV}/bin/python" "${SCRIPT_DIR}/romem_tkg_benchmark.py" \
        --dataset-dir "${ROMEM_DATA}" \
        --model romem \
        --embedding-dim 64 \
        --epochs ${ROMEM_EPOCHS} \
        --batch-size 512 \
        --lr 0.001 \
        --num-negatives 128 \
        --num-conflict-negatives 1 \
        --margin 0.5 \
        --validate-every ${ROMEM_VALID_FREQ} \
        --time-contrastive-weight 0.0 \
        2>&1 | tee "${RESULTS_DIR}/romem_no_tc.log"
}

run_romem_nogate() {
    log "RoMem-NoGate (rotation only, alpha=1) on ${ROMEM_DATA} (${USE_DEVICE})"
    activate_venv
    "${VENV}/bin/python" "${SCRIPT_DIR}/romem_tkg_benchmark.py" \
        --dataset-dir "${ROMEM_DATA}" \
        --model romem_nogate \
        --embedding-dim 64 \
        --epochs ${ROMEM_EPOCHS} \
        --batch-size 512 \
        --lr 0.001 \
        --num-negatives 128 \
        --num-conflict-negatives 1 \
        --margin 0.5 \
        --validate-every ${ROMEM_VALID_FREQ} \
        --time-contrastive-weight 0.5 \
        2>&1 | tee "${RESULTS_DIR}/romem_nogate.log"
}

run_chronor_romem() {
    log "ChronoR-RoMem (k=3, CE loss, functional rotation + gate) on ${ROMEM_DATA} (${USE_DEVICE})"
    activate_venv
    "${VENV}/bin/python" "${SCRIPT_DIR}/romem_tkg_benchmark.py" \
        --dataset-dir "${ROMEM_DATA}" \
        --model chronor_romem \
        --embedding-dim 500 --gamma 200.0 --k 3 \
        --batch-size 1000 --lr 0.1 --optimizer adagrad \
        --regularization 0.01 \
        --num-negatives 256 --adversarial-temperature 1.0 \
        --num-conflict-negatives 1 \
        --epochs ${ROMEM_EPOCHS} \
        --validate-every ${ROMEM_VALID_FREQ} \
        --time-contrastive-weight 0.5 \
        2>&1 | tee "${RESULTS_DIR}/chronor_romem.log"
}

run_chronor_romem_nogate() {
    log "ChronoR-RoMem-NoGate (k=3, CE loss, alpha=1, TC) on ${ROMEM_DATA} (${USE_DEVICE})"
    activate_venv
    "${VENV}/bin/python" "${SCRIPT_DIR}/romem_tkg_benchmark.py" \
        --dataset-dir "${ROMEM_DATA}" \
        --model chronor_romem_nogate \
        --embedding-dim 500 --gamma 200.0 --k 3 \
        --batch-size 1000 --lr 0.1 --optimizer adagrad \
        --regularization 0.01 \
        --num-negatives 256 --adversarial-temperature 1.0 \
        --num-conflict-negatives 1 \
        --epochs ${ROMEM_EPOCHS} \
        --validate-every ${ROMEM_VALID_FREQ} \
        --time-contrastive-weight 0.5 \
        2>&1 | tee "${RESULTS_DIR}/chronor_romem_nogate.log"
}

run_chronor_romem_notc() {
    log "ChronoR-RoMem-NoTC (k=3, CE loss, no time contrastive) on ${ROMEM_DATA} (${USE_DEVICE})"
    activate_venv
    "${VENV}/bin/python" "${SCRIPT_DIR}/romem_tkg_benchmark.py" \
        --dataset-dir "${ROMEM_DATA}" \
        --model chronor_romem_notc \
        --embedding-dim 500 --gamma 200.0 --k 3 \
        --batch-size 1000 --lr 0.1 --optimizer adagrad \
        --regularization 0.01 \
        --num-negatives 256 --adversarial-temperature 1.0 \
        --num-conflict-negatives 1 \
        --epochs ${ROMEM_EPOCHS} \
        --validate-every ${ROMEM_VALID_FREQ} \
        2>&1 | tee "${RESULTS_DIR}/chronor_romem_notc.log"
}

# ── summary extractor ───────────────────────────────────────────────────────
extract_results() {
    log "RESULTS SUMMARY"
    echo ""
    printf "%-15s  %8s  %8s  %8s  %8s\n" "Model" "MRR" "Hits@1" "Hits@3" "Hits@10"
    printf "%-15s  %8s  %8s  %8s  %8s\n" "───────────────" "────────" "────────" "────────" "────────"

    # DE-SimplE
    if [ -f "${RESULTS_DIR}/de_simple.log" ]; then
        python3 -c "
import re, sys
text = open('${RESULTS_DIR}/de_simple.log').read()
blocks = text.split('Fil setting:')
if len(blocks) >= 2:
    block = blocks[-1]
    mrr = re.search(r'MRR\s*=\s*([\d.]+)', block)
    h1  = re.search(r'Hit@1\s*=\s*([\d.]+)', block)
    h3  = re.search(r'Hit@3\s*=\s*([\d.]+)', block)
    h10 = re.search(r'Hit@10\s*=\s*([\d.]+)', block)
    if mrr:
        print(f'DE-SimplE        {float(mrr.group(1)):8.4f}  {float(h1.group(1)):8.4f}  {float(h3.group(1)):8.4f}  {float(h10.group(1)):8.4f}')
    else:
        print('DE-SimplE        (parse error)')
else:
    print('DE-SimplE        (not found)')
" 2>/dev/null || echo "DE-SimplE        (log not found)"
    fi

    # ChronoR
    if [ -f "${RESULTS_DIR}/chronor.log" ]; then
        python3 -c "
import re
text = open('${RESULTS_DIR}/chronor.log').read()
m = re.findall(r\"TEST\s*:\s*\{'MRR':\s*([\d.]+),\s*'hits@\[1,3,10\]':\s*tensor\(\[([\d.,\s]+)\]\)\", text)
if m:
    mrr = float(m[-1][0])
    hits = [float(x.strip()) for x in m[-1][1].split(',')]
    print(f'ChronoR          {mrr:8.4f}  {hits[0]:8.4f}  {hits[1]:8.4f}  {hits[2]:8.4f}')
else:
    print('ChronoR          (parse error)')
" 2>/dev/null || echo "ChronoR          (log not found)"
    fi

    # RotatE
    if [ -f "${RESULTS_DIR}/rotate.log" ]; then
        python3 -c "
import re
text = open('${RESULTS_DIR}/rotate.log').read()
lines = text.strip().split('\n')
metrics = {}
for line in reversed(lines):
    if 'Test' in line and 'at step' in line:
        m = re.search(r'Test\s+(\S+)\s+at step\s+\d+:\s+([\d.]+)', line)
        if m and m.group(1) not in metrics:
            metrics[m.group(1)] = float(m.group(2))
if 'MRR' in metrics:
    print(f\"RotatE           {metrics.get('MRR',0):8.4f}  {metrics.get('HITS@1',0):8.4f}  {metrics.get('HITS@3',0):8.4f}  {metrics.get('HITS@10',0):8.4f}\")
else:
    print('RotatE           (parse error)')
" 2>/dev/null || echo "RotatE           (log not found)"
    fi

    # DistMult
    if [ -f "${RESULTS_DIR}/distmult.log" ]; then
        python3 -c "
import re
text = open('${RESULTS_DIR}/distmult.log').read()
lines = text.strip().split('\n')
metrics = {}
for line in reversed(lines):
    if 'Test' in line and 'at step' in line:
        m = re.search(r'Test\s+(\S+)\s+at step\s+\d+:\s+([\d.]+)', line)
        if m and m.group(1) not in metrics:
            metrics[m.group(1)] = float(m.group(2))
if 'MRR' in metrics:
    print(f\"DistMult         {metrics.get('MRR',0):8.4f}  {metrics.get('HITS@1',0):8.4f}  {metrics.get('HITS@3',0):8.4f}  {metrics.get('HITS@10',0):8.4f}\")
else:
    print('DistMult         (parse error)')
" 2>/dev/null || echo "DistMult         (log not found)"
    fi

    # RoMem models (all share the same log format)
    for romem_entry in "romem:RoMem" "romem_no_tc:RoMem-NoTC" "romem_nogate:RoMem-NoGate" "chronor_romem:ChronoR-RoMem" "chronor_romem_nogate:ChronoR-RoMem-NG" "chronor_romem_notc:ChronoR-RoMem-NoTC"; do
        logname="${romem_entry%%:*}"
        label="${romem_entry##*:}"
        if [ -f "${RESULTS_DIR}/${logname}.log" ]; then
            python3 -c "
import re
name = '${label}'
text = open('${RESULTS_DIR}/${logname}.log').read()
blocks = text.split('Fil setting:')
if len(blocks) >= 2:
    block = blocks[-1]
    mrr = re.search(r'MRR\s*=\s*([\d.]+)', block)
    h1  = re.search(r'Hit@1\s*=\s*([\d.]+)', block)
    h3  = re.search(r'Hit@3\s*=\s*([\d.]+)', block)
    h10 = re.search(r'Hit@10\s*=\s*([\d.]+)', block)
    if mrr:
        print(f'{name:<15s}  {float(mrr.group(1)):8.4f}  {float(h1.group(1)):8.4f}  {float(h3.group(1)):8.4f}  {float(h10.group(1)):8.4f}')
    else:
        print(f'{name:<15s}  (parse error)')
else:
    print(f'{name:<15s}  (not found)')
" 2>/dev/null || echo "${label}  (log not found)"
        fi
    done

    echo ""
}

# ── main ─────────────────────────────────────────────────────────────────────
ALL_MODELS="de_simple chronor rotate distmult romem romem_no_tc romem_nogate chronor_romem chronor_romem_nogate chronor_romem_notc"

if [ $# -eq 0 ]; then
    MODELS="${ALL_MODELS}"
else
    MODELS=""
    for arg in "$@"; do
        arg_lower="$(echo "$arg" | tr '[:upper:]' '[:lower:]')"
        case "${arg_lower}" in
            de_simple|de-simple|desimple)   MODELS+=" de_simple" ;;
            chronor)                        MODELS+=" chronor" ;;
            rotate)                         MODELS+=" rotate" ;;
            distmult)                       MODELS+=" distmult" ;;
            romem)                          MODELS+=" romem" ;;
            romem_no_tc|romem-no-tc)        MODELS+=" romem_no_tc" ;;
            romem_nogate|romem-nogate)      MODELS+=" romem_nogate" ;;
            chronor_romem|chronor-romem)    MODELS+=" chronor_romem" ;;
            chronor_romem_nogate|chronor-romem-nogate)  MODELS+=" chronor_romem_nogate" ;;
            chronor_romem_notc|chronor-romem-notc)      MODELS+=" chronor_romem_notc" ;;
            all)                            MODELS="${ALL_MODELS}" ;;
            summary)                        extract_results; exit 0 ;;
            *)  echo "Unknown model: $arg"; echo "Available: de_simple chronor rotate distmult romem romem_no_tc romem_nogate chronor_romem chronor_romem_nogate chronor_romem_notc all summary"; exit 1 ;;
        esac
    done
fi

# ── auto-prepare subset data if needed ───────────────────────────────────────
if [ "${SUBSET}" = "1" ]; then
    if [ ! -f "${SCRIPT_DIR}/de-simple-master/datasets/icews05-15-sub/train.txt" ]; then
        echo "Preparing 20K-triple subset..."
        activate_venv
        python "${SCRIPT_DIR}/prepare_subset.py" --max-train 20000
    fi
fi

echo "Models to benchmark: ${MODELS}"
echo "Device: ${USE_DEVICE}"
if [ "${QUICK}" = "1" ]; then
    echo "Mode: QUICK (reduced epochs for sanity check)"
fi
if [ "${SUBSET}" = "1" ]; then
    echo "Data: SUBSET (20K train triples)"
fi
echo "Results: ${RESULTS_DIR}/"
echo ""



extract_results
