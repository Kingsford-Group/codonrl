## Checkpoints

We provide 55 model checkpoints trained on different protein sequences from the UniProt dataset. 
### Download Options

####  Download All Checkpoints

Download all 55 checkpoints using the provided script:

```bash
# Download the checkpoint list
wget https://datarnadesign.blob.core.windows.net/codonrl-checkpoints/checkpoint_urls.txt
```

```bash
# Download all checkpoints
mkdir -p checkpoints
cd checkpoints
while read url; do
    wget "$url"
done < ../checkpoint_urls.txt

```


## Training

### Basic Usage

```bash
# Batch training from JSON file
python CodonRL_main.py --jf datasets/proteins.json
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--jf` / `--json_input_file` | Required | Training dataset (JSON format) |
| `--codon_table` | `human` | Codon table: `human` or `ecolik12` |
| `--lambda_val` | `4.0` | CAI-MFE tradeoff (0=MFE only, 10=CAI priority) |
| `--protein_max_len` | `700` | Maximum protein length to process |
| `--batch_size` | `64` | Training batch size |
| `-e` / `--num_episodes` | `250` | Total training episodes |
| `--learning_rate` | `5e-5` | Learning rate |
| `--buffer_size` | `10000` | Experience replay buffer size |
| `--target_update_freq` | `50` | Target network update frequency (steps) |
| `--max_workers` | `None` | Max parallel workers (auto-detected if None) |
| `--mfe_workers` | `4` | Thread-pool size per process for MFE calculations |
| `--milestone_mfe_method` | `linearfold` | MFE method during training: `linearfold` or `vienna` |
| `--final_mfe_method` | `vienna` | MFE method for final evaluation |
| `--output_dir` | `results` | Directory for checkpoints and logs |

### GPU Configuration

**Single GPU (50 workers):**
```bash
export DEVICES=$(python3 -c "print(','.join(['cuda:0']*50))")
```

**Multi-GPU (e.g., 4 GPUs with 12-13 workers each):**
```bash
export DEVICES=$(python3 -c "
devices = []
for i in range(4):
    devices.extend([f'cuda:{i}']*13)
print(','.join(devices[:50]))
")
```

**Manual configuration:**
```bash
export DEVICES="cuda:0,cuda:0,cuda:1,cuda:1,cuda:2,cuda:2,cuda:3,cuda:3"
```

### Optional Flags

```bash
--use_amp                    # Enable automatic mixed precision (recommended for modern GPUs)
--prepopulate_buffer         # Pre-fill replay buffer before training starts
--wandb_log                  # Enable Weights & Biases logging
--wandb_project <name>       # W&B project name
--wandb_run_name_prefix <p>  # W&B run name prefix for experiment tracking
```


### Example Configurations


#### 
```bash
DEVICES=$(python3 -c "print(','.join(['cuda:0']*50))")

nohup python CodonRL_main.py -jf ./datasets/uniprot_le_500/uniprot_with_guidance_l0.json \
  --codon_table human \
  --wandb_log \
  --lambda_val 4 \
  --protein_max_len 501 \
  --prepopulate_buffer \
  --batch_size 64 \
  -e 500 \
  --buffer_size 100000 \
  --wandb_project CodonRl \
  --use_amp \
  --target_update_freq 150 \
  --learning_rate 2e-5 \
  --wandb_run_name_prefix setting1 \
  --parallel_devices $DEVICES\
  --max_workers 55 \
  --mfe_workers 4 \
  --milestone_mfe_method linearfold \
  --final_mfe_method linearfold \
  --output_dir results
```
