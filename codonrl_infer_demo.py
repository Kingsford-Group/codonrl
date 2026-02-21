#!/usr/bin/env python3
"""
CodonRL inference demo (single-file)

What it does
- Loads CodonRL config from a training_summary.json
- Loads model weights from ckpt_best_objective.pth (or a user-specified ckpt path)
- Takes a protein sequence as input
- Decodes an mRNA sequence using the same hybrid_decode() logic in visualizeandbenchmark.py
- Writes the CodonRL output to FASTA:
    <out_prefix>_codonrl_rna.fasta
    <out_prefix>_codonrl_dna.fasta

Notes
- This script assumes the CodonRL code is importable as `CodonRL_main`.
  If it is not installed as a package, pass --codonrl_path to add it to PYTHONPATH.
- MFE is (optionally) computed after decoding, using get_mfe_calculator() from CodonRL_main.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional, Tuple

import torch

# -------------------------------------------------------------------
# Optional: add codonrl source path so `from CodonRL_main import ...` works
# -------------------------------------------------------------------
def maybe_add_codonrl_path(codonrl_path: Optional[str]) -> None:
    if codonrl_path:
        codonrl_path = os.path.abspath(codonrl_path)
        if codonrl_path not in sys.path:
            sys.path.insert(0, codonrl_path)

# We import after maybe_add_codonrl_path() in main()

# -------------------------------------------------------------------
# Helpers copied/adapted from visualizeandbenchmark.py
# -------------------------------------------------------------------
def to_dna(seq: str) -> str:
    return (seq or "").strip().upper().replace("U", "T")

def to_rna(seq: str) -> str:
    return (seq or "").strip().upper().replace("T", "U")

def load_cfg_and_w(summary_json_path: str):
    """
    Load cfg from training_summary.json and compute CAI relative adaptiveness table w.
    """
    with open(summary_json_path, "r") as f:
        summ = json.load(f)
    cfg = summ["config"]

    table = (cfg.get("codon_table", "human") or "human").lower()
    if table == "human":
        freq = HUMAN_FREQ_PER_THOUSAND
    elif table in ("ecolik12", "ecoli", "ecolik-12", "e.coli"):
        freq = ECOLLI_K12_FREQ_PER_THOUSAND
    else:
        raise ValueError(f"Unsupported codon table in cfg: {table}")

    w = calculate_relative_adaptiveness(AA_TO_CODONS, freq)
    configure_target_w_table(w)
    return cfg, w

def build_agent(cfg: dict, device: str):
    """
    Build CodonRL agent with exploration disabled for inference.
    """
    cfg = dict(cfg)
    cfg["device"] = device
    cfg["use_amp"] = False
    cfg["eps_start"] = 0.0
    cfg["eps_end"] = 0.0
    cfg["eps_decay"] = 1
    return CodonRL(cfg)

def hybrid_decode(agent, protein: str, w: dict, alpha: float = 0.5) -> str:
    """
    Greedy decoding with an inference-time CAI bias term alpha * log(w[codon]),
    matching visualizeandbenchmark.py.
    """
    logw = {c: (math.log(max(w.get(c, 1e-12), 1e-12))) for c in w}

    agent._precompute_protein_memory(protein)
    mrna = ""
    for t, aa in enumerate(protein):
        state = agent._get_state(mrna, t)
        with torch.no_grad():
            q = agent.policy_net.decode_mrna(
                state["mrna"],
                state["pos"],
                agent.protein_memory_cache,
                agent.protein_pad_mask_cache,
            )[0]
            idxs = [CODON_TO_INT[c] for c in AA_TO_CODONS[aa]]
            scores = [(q[i].item() + alpha * logw[INT_TO_CODON[i]], i) for i in idxs]
            _, i_best = max(scores, key=lambda x: x[0])
        mrna += INT_TO_CODON[i_best]
    return mrna

def validate_protein_seq(protein: str) -> str:
    protein = (protein or "").strip().upper()
    if not protein:
        raise ValueError("Empty protein sequence.")
    bad = [aa for aa in protein if aa not in AA_TO_CODONS]
    if bad:
        bad_set = "".join(sorted(set(bad)))
        raise ValueError(f"Protein contains unsupported amino acids: {bad_set}")
    return protein

def write_fasta(path: str, seq_id: str, seq: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f">{seq_id}\n{seq}\n")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CodonRL single-sequence inference demo (writes FASTA).")
    p.add_argument("--protein_seq", type=str, default=None,
                   help="Protein sequence (amino acids). If omitted, use --protein_fasta.")
    p.add_argument("--protein_fasta", type=str, default=None,
                   help="FASTA file containing a single protein sequence.")
    p.add_argument("--seq_id", type=str, default="query",
                   help="Sequence ID for FASTA header.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Inference-time CAI bias coefficient used in hybrid_decode.")
    p.add_argument("--summary_json", type=str, required=True,
                   help="Path to training_summary.json (contains cfg).")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Path to checkpoint .pth (default: <ckpt_dir>/ckpt_best_objective.pth).")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="Directory containing ckpt_best_objective.pth (used if --ckpt_path not set).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device string, e.g. cpu or cuda:0.")
    p.add_argument("--out_prefix", type=str, default="./codonrl_output",
                   help="Output prefix (writes *_codonrl_rna.fasta and *_codonrl_dna.fasta).")
    p.add_argument("--codonrl_path", type=str, default=os.getenv("CODONRL_PATH"),
                   help="Optional path to CodonRL source directory (added to PYTHONPATH).")
    p.add_argument("--compute_mfe", action="store_true",
                   help="If set, compute MFE after decoding (Vienna preferred, else LinearFold) and print it.")
    p.add_argument("--compute_cai", action="store_true",
                   help="If set, compute CAI after decoding and print it.")

    args = p.parse_args()

    # Make codonrl importable
    maybe_add_codonrl_path(args.codonrl_path)

    # Import CodonRL symbols (must come after sys.path update)
    global CodonRL, configure_target_w_table, calculate_relative_adaptiveness
    global HUMAN_FREQ_PER_THOUSAND, ECOLLI_K12_FREQ_PER_THOUSAND
    global AA_TO_CODONS, CODON_TO_INT, INT_TO_CODON, calculate_cai, get_mfe_calculator
    try:
        from CodonRL_main import (
            CodonRL,
            configure_target_w_table,
            calculate_relative_adaptiveness,
            HUMAN_FREQ_PER_THOUSAND,
            ECOLLI_K12_FREQ_PER_THOUSAND,
            AA_TO_CODONS,
            CODON_TO_INT,
            INT_TO_CODON,
            calculate_cai,
            get_mfe_calculator,
        )
    except Exception as e:
        raise SystemExit(
            "Failed to import CodonRL_main. "
            "If CodonRL is not installed as a package, pass --codonrl_path /path/to/codonrl.\n"
            f"Import error: {e}"
        )

    # Read protein sequence
    protein = args.protein_seq
    if protein is None and args.protein_fasta:
        with open(args.protein_fasta, "r") as f:
            seq_lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    continue
                seq_lines.append(line)
        protein = "".join(seq_lines)

    protein = validate_protein_seq(protein)

    # Determine checkpoint path
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        if not args.ckpt_dir:
            raise SystemExit("Provide either --ckpt_path or --ckpt_dir.")
        ckpt_path = os.path.join(args.ckpt_dir, "ckpt_best_objective.pth")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(args.summary_json):
        raise SystemExit(f"training_summary.json not found: {args.summary_json}")

    # Load cfg + CAI weights w
    cfg, w = load_cfg_and_w(args.summary_json)

    # Build agent and load weights
    agent = build_agent(cfg, args.device)
    sd = torch.load(ckpt_path, map_location=args.device)
    agent.policy_net.load_state_dict(sd)
    agent.target_net.load_state_dict(sd)

    # Decode mRNA
    mrna = hybrid_decode(agent, protein, w, alpha=args.alpha)
    mrna_rna = to_rna(mrna)
    mrna_dna = to_dna(mrna)

    # Write FASTA outputs
    out_rna = f"{args.out_prefix}_codonrl_rna.fasta"
    out_dna = f"{args.out_prefix}_codonrl_dna.fasta"
    write_fasta(out_rna, args.seq_id, mrna_rna)
    write_fasta(out_dna, args.seq_id, mrna_dna)

    print(f"[CodonRL] Wrote RNA FASTA: {out_rna}")
    print(f"[CodonRL] Wrote DNA FASTA: {out_dna}")
    print(f"[CodonRL] Protein length: {len(protein)} aa; mRNA length: {len(mrna_rna)} nt")

    # Optional metrics (post-hoc)
    if args.compute_cai:
        # Ensure CAI sees the right alphabet; w is keyed by codons (DNA or RNA depending on table)
        # We attempt both and pick the one that works.
        cai = None
        try:
            cai = calculate_cai(mrna_rna, w)
        except Exception:
            cai = calculate_cai(mrna_dna, w)
        print(f"[CodonRL] CAI: {cai:.6f}")

    if args.compute_mfe:
        mfe_calc = get_mfe_calculator()
        # Prefer Vienna, fallback to LinearFold
        vienna_mfe = mfe_calc.calculate_vienna_async(mrna_rna).result()
        lf_mfe = mfe_calc.calculate_linearfold_async(mrna_rna).result()
        mfe = vienna_mfe if math.isfinite(vienna_mfe) else lf_mfe
        src = "ViennaRNA" if math.isfinite(vienna_mfe) else "LinearFold"
        print(f"[CodonRL] MFE ({src}): {mfe:.6f}")

if __name__ == "__main__":
    main()
