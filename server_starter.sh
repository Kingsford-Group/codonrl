export SUMMARY_JSON="checkpoints/1_linearfold_linearfold/training_summary.json"
export CKPT_DIR="checkpoints/1_linearfold_linearfold/"          # contains ckpt_best_objective.pth
# OR instead of CKPT_DIR:
# export CKPT_PATH="/path/to/ckpt_best_objective.pth"

# If CodonRL_main isn't importable normally:
#export CODONRL_PATH="/path/to/codonrl_source"

export DEVICE="cpu"        # or cuda:0
export ALPHA="0.5"


python3 codonrl_demo_app.py
