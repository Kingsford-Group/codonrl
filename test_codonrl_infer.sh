#!/bin/bash

python3 codonrl_infer_demo.py \
  --protein_fasta demo_protein.fasta \
  --summary_json checkpoints/1_linearfold_linearfold/training_summary.json \
  --ckpt_dir checkpoints/1_linearfold_linearfold/ \
  --alpha 2.0 \
  --out_prefix demo_out/run1 \
  --seq_id my_rna \
  --compute_cai --compute_mfe