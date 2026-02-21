#!/usr/bin/env python3
"""
CodonRL demo backend (FastAPI)

What you get
- GET  /              -> serves a simple centered HTML demo page
- POST /infer         -> accepts a protein FASTA upload, runs CodonRL inference, returns JSON:
                         {id, protein_length, rna_fasta, dna_fasta, rna_seq, dna_seq}

Configuration (via environment variables)
- CODONRL_PATH   : path to your CodonRL source directory (if not installed as a package)
- SUMMARY_JSON   : path to training_summary.json
- CKPT_PATH      : (optional) path to checkpoint .pth
- CKPT_DIR       : (optional) directory containing ckpt_best_objective.pth (used if CKPT_PATH not set)
- DEVICE         : torch device string, e.g. "cpu" or "cuda:0" (default "cpu")
- ALPHA          : default decoding CAI-bias coefficient (default 0.5)

Run
  python codonrl_demo_app.py
or
  uvicorn codonrl_demo_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import math
import os
import sys
from typing import Tuple

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------
# Config
# ---------------------------
CODONRL_PATH = os.getenv("CODONRL_PATH")  # optional
SUMMARY_JSON = os.getenv("SUMMARY_JSON")  # required
CKPT_PATH = os.getenv("CKPT_PATH")        # optional
CKPT_DIR = os.getenv("CKPT_DIR")          # optional
DEVICE = os.getenv("DEVICE", "cpu")
DEFAULT_ALPHA = float(os.getenv("ALPHA", "0.5"))

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "codonrl_demo.html")

# ---------------------------
# Make CodonRL importable
# ---------------------------
if CODONRL_PATH:
    CODONRL_PATH = os.path.abspath(CODONRL_PATH)
    if CODONRL_PATH not in sys.path:
        sys.path.insert(0, CODONRL_PATH)

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
        "Failed to import CodonRL_main. Set CODONRL_PATH or install your package.\n"
        f"Import error: {e}"
    )

# ---------------------------
# FASTA + misc utilities
# ---------------------------
def parse_single_fasta(text: str) -> Tuple[str, str]:
    """Parse a FASTA containing ONE protein sequence."""
    header = "protein_from_fasta"
    seq_parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header == "protein_from_fasta":
                header = line[1:].strip() or header
            continue
        seq_parts.append(line)
    seq = "".join(seq_parts).upper().replace(" ", "").replace("\t", "")
    if not seq:
        raise ValueError("No sequence found in FASTA.")
    return header, seq

def validate_protein_seq(protein: str) -> str:
    protein = (protein or "").strip().upper()
    if not protein:
        raise ValueError("Empty protein sequence.")
    bad = [aa for aa in protein if aa not in AA_TO_CODONS]
    if bad:
        bad_set = "".join(sorted(set(bad)))
        raise ValueError(f"Unsupported amino-acid letter(s): {bad_set}")
    return protein

def to_rna(seq: str) -> str:
    return (seq or "").upper().replace("T", "U")

def to_dna(seq: str) -> str:
    return (seq or "").upper().replace("U", "T")

def wrap_fasta(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def format_fasta(seq_id: str, seq: str) -> str:
    return f">{seq_id}\n{wrap_fasta(seq)}\n"

# ---------------------------
# Inference logic (mirrors visualizeandbenchmark.py usage)
# ---------------------------
def load_cfg_and_w(summary_json_path: str):
    import json
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
    cfg = dict(cfg)
    cfg["device"] = device
    cfg["use_amp"] = False
    cfg["eps_start"] = 0.0
    cfg["eps_end"] = 0.0
    cfg["eps_decay"] = 1
    return CodonRL(cfg)

def hybrid_decode(agent, protein: str, w: dict, alpha: float = 0.5) -> str:
    logw = {c: math.log(max(w.get(c, 1e-12), 1e-12)) for c in w}
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

def resolve_ckpt_path() -> str:
    if CKPT_PATH:
        return CKPT_PATH
    if CKPT_DIR:
        return os.path.join(CKPT_DIR, "ckpt_best_objective.pth")
    raise SystemExit("Set CKPT_PATH or CKPT_DIR environment variable.")

if not SUMMARY_JSON:
    raise SystemExit("Set SUMMARY_JSON environment variable to training_summary.json")

_ckpt_path = resolve_ckpt_path()
if not os.path.exists(SUMMARY_JSON):
    raise SystemExit(f"SUMMARY_JSON not found: {SUMMARY_JSON}")
if not os.path.exists(_ckpt_path):
    raise SystemExit(f"Checkpoint not found: {_ckpt_path}")

_cfg, _w = load_cfg_and_w(SUMMARY_JSON)
_agent = build_agent(_cfg, DEVICE)
_sd = torch.load(_ckpt_path, map_location=DEVICE)
_agent.policy_net.load_state_dict(_sd)
_agent.target_net.load_state_dict(_sd)

# ---------------------------
# API
# ---------------------------
app = FastAPI(title="CodonRL inference demo")

@app.get("/")
def root():
    if not os.path.exists(HTML_PATH):
        return JSONResponse({"error": "codonrl_demo.html not found next to codonrl_demo_app.py"}, status_code=500)
    return FileResponse(HTML_PATH, media_type="text/html")

@app.post("/infer")
async def infer(
    protein_fasta: UploadFile = File(...),
    alpha: float = Query(DEFAULT_ALPHA, description="CAI-bias coefficient used during decoding."),
    include_metrics: bool = Query(False, description="If true, also compute CAI and MFE (post-hoc)."),
):
    try:
        text = (await protein_fasta.read()).decode("utf-8", errors="replace")
        seq_id, protein = parse_single_fasta(text)
        protein = validate_protein_seq(protein)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    mrna_codons = hybrid_decode(_agent, protein, _w, alpha=alpha)
    rna_seq = to_rna(mrna_codons)
    dna_seq = to_dna(mrna_codons)

    payload = {
        "id": seq_id,
        "protein_length": len(protein),
        "rna_seq": rna_seq,
        "rna_fasta": format_fasta(seq_id, rna_seq),
        "alpha": alpha,
    }

    if include_metrics:
        try:
            payload["cai"] = float(calculate_cai(rna_seq, _w))
        except Exception:
            payload["cai"] = "CAI calculation is not available due to error."

        mfe_calc = get_mfe_calculator()
        vienna_mfe = mfe_calc.calculate_vienna_async(rna_seq).result()
        lf_mfe = mfe_calc.calculate_linearfold_async(rna_seq).result()
        mfe = float(vienna_mfe) if math.isfinite(vienna_mfe) else float(lf_mfe)
        payload["mfe"] = mfe
        payload["mfe_source"] = "ViennaRNA" if math.isfinite(vienna_mfe) else "LinearFold"

    return JSONResponse(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
