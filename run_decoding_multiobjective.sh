#!/bin/bash

OUT_DIR=./benchmark_multialpha
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

alphas=(1.0 1.5 2.0 2.5 3.0)


# below w except w_cai are kind of randomly selected
for alpha in "${alphas[@]}"; do
  echo "Starting benchmark with alpha=${alpha} in background..."

  nohup python ./visualizeandbenchmark.py \
    --csv ./datasets/gemorna_with_all_metrics.csv \
    --ckpt_root ./checkpoints \
    --alpha ${alpha} \
    --w_cai 1.0 --w_mfe 0.05 --w_csc 0.30 --w_gc 0.10 --w_u 0.05 \
    --outdir "$OUT_DIR" \
    --run_name gemorna_multi_balanced-alpha${alpha} \
    --title "Multi-metric (balanced) alpha=${alpha}" \
    --csc_file ./config/csc.json \
    > "$LOG_DIR/multiobjective_alpha${alpha}.log" 2>&1 &

  echo "Started alpha=${alpha} (PID: $!)"
done

echo ""
echo "All benchmarks started in background!"
echo "Monitor progress with: tail -f $LOG_DIR/multiobjective_alpha*.log"
echo "Check running processes with: ps aux | grep visualizeandbenchmark"



