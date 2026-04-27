#!/usr/bin/env bash
set -euo pipefail

PARALLEL_DEVICES=$(printf 'cuda:0,%.0s' {1..50})
PARALLEL_DEVICES=${PARALLEL_DEVICES%,}

python CodonRL_main.py \
    -jf ../../datasets/uniprot_le_500/uniprot_with_guidance_l0.json \
    --codon_table human \
    --wandb_log \
    --lambda_val 4 \
    --protein_max_len 501 \
    --prepopulate_buffer \
    --batch_size 64 \
    -e 500 \
    --buffer_size 100000 \
    --wandb_project TBD \
    --use_amp \
    --target_update_freq 150 \
    --learning_rate 2e-5 \
    --wandb_run_name_prefix TBD\
    --parallel_devices "$PARALLEL_DEVICES" \
    --max_workers 55 \
    --mfe_workers 4 \
    --milestone_mfe_method linearfold \
    --final_mfe_method vienna \
    --output_dir results
