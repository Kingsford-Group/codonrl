#!/bin/bash

# ==============================================================================
# Multi-objective mRNA optimization benchmark runner
# ==============================================================================

OUT_DIR=./benchmark_multialpha
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# ==============================================================================
# Configuration
# ==============================================================================

CSV_PATH=./datasets/gemorna_with_all_metrics.csv
CKPT_ROOT=./checkpoints
CSC_FILE=./config/csc.json
SCRIPT_PATH=./visualizeandbenchmark_multialpha.py


# ==============================================================================
# U content optimization
# ==============================================================================
echo "========================================"
echo " U content optimization"
echo "========================================"

# Test different U weights (negative to minimize U content)
alpha_cai=2.5
# alphas_u=(-0.1 -0.2 -0.3 -0.4 -0.5)
alphas_u=(0.0 0.1 0.2 0.3 0.4 0.5)

for alpha_u in "${alphas_u[@]}"; do
  echo "Starting U optimization benchmark with alpha_u=${alpha_u}..."

  nohup python "$SCRIPT_PATH" \
    --csv "$CSV_PATH" \
    --ckpt_root "$CKPT_ROOT" \
    --alpha_cai ${alpha_cai} \
    --alpha_csc 0.0 \
    --alpha_gc 0.0 \
    --alpha_u ${alpha_u} \
    --w_cai 1.0 \
    --w_mfe 1.0 \
    --w_csc 0.0 \
    --w_gc 0.0 \
    --w_u -1.0 \
    --csc_file "$CSC_FILE" \
    --outdir "$OUT_DIR" \
    --run_name "u_optimization_au${alpha_u}" \
    --title "U Minimization (α_CAI=2.5, α_U=${alpha_u})" \
    > "$LOG_DIR/exp3_u_opt_au${alpha_u}.log" 2>&1 &

  echo "  Started alpha_u=${alpha_u} (PID: $!)"
done

echo ""
