#!/bin/bash

python soup55_rolling_inference.py \
    --protein_file ./datasets/examples/Q3L8U1_gemorna.fasta \
    --gemorna_file ./datasets/examples/Q3L8U1.fasta \
    --soup_checkpoint ./model_soup_checkpoint/soup55.pth \
    --output soup55_rolling_Q3L8U1.json \
    --window_size 500 \
    --alpha_cai 2.5 \
    --alpha_csc 0 \
    --alpha_gc 0 \
    --alpha_u 1
