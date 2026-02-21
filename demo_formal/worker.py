# worker.py
import os
import sys
import math
import time
import traceback
from multiprocessing import current_process

import requests
import torch

# ====== configure our CodonRL import path ======
# Option 1: export CODONRL_PATH=/abs/path/to/codonrl
CODONRL_PATH = os.environ.get("CODONRL_PATH", None)
if CODONRL_PATH:
    sys.path.append(CODONRL_PATH)
else:
    # Option 2: hardcode like our benchmark script did
    sys.path.append("/path/to/codonrl")  # TODO change me

# ====== imports from CodonRL repo (same as our benchmark script) ======
from CodonRL_main import (
    CodonRL, configure_target_w_table, calculate_relative_adaptiveness,
    HUMAN_FREQ_PER_THOUSAND, ECOLLI_K12_FREQ_PER_THOUSAND,
    AA_TO_CODONS, CODON_TO_INT, calculate_cai, get_mfe_calculator
)

# -----------------------------
# Helpers (taken from our benchmark script)
# -----------------------------
def calculate_codon_gc(codon: str) -> float:
    return (codon.count("G") + codon.count("C")) / 3.0

def calculate_codon_u(codon: str) -> float:
    codon_rna = codon.replace("T", "U")
    return codon_rna.count("U") / 3.0

def gc_content(seq: str) -> float:
    s = (seq or "").strip().upper()
    if not s:
        return math.nan
    return (s.count("G") + s.count("C")) / len(s)

def u_pct(seq: str) -> float:
    s = (seq or "").strip().upper()
    if not s:
        return math.nan
    return (s.count("U") + s.count("T")) / len(s)

def load_cfg_and_w(cfg: dict):
    """
    In our benchmark, cfg came from training_summary.json.
    Here, we assume cfg already provided by the caller (server).
    """
    cfg = dict(cfg)
    table = cfg.get("codon_table", "human").lower()
    if table == "human":
        freq = HUMAN_FREQ_PER_THOUSAND
    elif table in ("ecolik12", "ecoli", "ecolik-12", "e.coli"):
        freq = ECOLLI_K12_FREQ_PER_THOUSAND
    else:
        raise ValueError(f"Unsupported codon table: {table}")

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

# -----------------------------
# EXTRACTED inference procedure (from our code in visualizeandbenchmark_multialpha.py)
# -----------------------------
def multiobjective_decode(
    agent,
    protein: str,
    w: dict,
    csc_weights: dict,
    *,
    alpha_cai: float = 0.5,
    alpha_csc: float = 0.0,
    alpha_gc: float = 0.0,
    alpha_u: float = 0.0,
    target_gc: float | None = None,
    target_u: float | None = None,
    mfe_calc=None,
    alpha_mfe: float = 0.0,
):
    """
    Extracted from visualizeandbenchmark_multialpha.py: multiobjective_decode(...) :contentReference[oaicite:2]{index=2}

    Greedy decoding over protein positions:
      - cache protein encoder once
      - at each step t: compute Q(s_t, codon) for candidate synonymous codons
      - add objective terms (CAI/CSC/GC/U/MFE)
      - pick best codon, append to prefix
    """
    logw = {c: math.log(max(w.get(c, 1e-12), 1e-12)) for c in w}
    logcsc = {c: math.log(max(csc_weights.get(c, 1e-12), 1e-12)) for c in csc_weights}

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

            candidate_codons = AA_TO_CODONS[aa]
            scores = []

            for codon in candidate_codons:
                idx = CODON_TO_INT[codon]
                score = q[idx].item()

                # CAI term
                if alpha_cai != 0:
                    score += alpha_cai * logw.get(codon, 0)

                # CSC term
                if alpha_csc != 0:
                    score += alpha_csc * logcsc.get(codon, 0)

                # GC term
                if alpha_gc != 0:
                    codon_gc = calculate_codon_gc(codon)
                    if target_gc is not None:
                        current_gc = gc_content(mrna) if mrna else 0.5
                        new_gc = (current_gc * (len(mrna) / 3) + codon_gc) / ((len(mrna) / 3) + 1)
                        gc_improvement = -abs(new_gc - target_gc)
                        score += alpha_gc * gc_improvement * 10
                    else:
                        score += alpha_gc * codon_gc

                # U term
                if alpha_u != 0:
                    codon_u = calculate_codon_u(codon)
                    if target_u is not None:
                        current_u = u_pct(mrna) if mrna else 0.25
                        new_u = (current_u * (len(mrna) / 3) + codon_u) / ((len(mrna) / 3) + 1)
                        u_improvement = -abs(new_u - target_u)
                        score += alpha_u * u_improvement * 10
                    else:
                        score += alpha_u * (-codon_u)

                # optional MFE term (slow)
                if alpha_mfe != 0 and mfe_calc is not None and len(mrna) >= 30:
                    test_mrna = mrna + codon
                    try:
                        mfe = mfe_calc.calculate_vienna_async(test_mrna).result(timeout=0.1)
                        if mfe is not None:
                            score += alpha_mfe * mfe
                    except Exception:
                        pass

                scores.append((score, codon))

            best_score, best_codon = max(scores, key=lambda x: x[0])
            mrna += best_codon

    return mrna

# -----------------------------
# Worker entry: run one job and report back to server
# -----------------------------
def run_job(
    *,
    server_base_url: str,
    job_id: str,
    protein: str,
    cfg: dict,
    ckpt_path: str,
    csc_weights: dict,
    decode_params: dict,
):
    """
    This is the "dockerized service" entrypoint for the demo:
      - loads agent + checkpoint
      - runs multiobjective_decode(...)
      - reports completion to server
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg, w = load_cfg_and_w(cfg)
        agent = build_agent(cfg, device)

        sd = torch.load(ckpt_path, map_location=device)
        agent.policy_net.load_state_dict(sd)
        agent.target_net.load_state_dict(sd)

        mfe_calc = get_mfe_calculator() if decode_params.get("alpha_mfe", 0.0) != 0.0 else None

        mrna = multiobjective_decode(
            agent=agent,
            protein=protein,
            w=w,
            csc_weights=csc_weights,
            alpha_cai=float(decode_params.get("alpha_cai", 0.5)),
            alpha_csc=float(decode_params.get("alpha_csc", 0.0)),
            alpha_gc=float(decode_params.get("alpha_gc", 0.0)),
            alpha_u=float(decode_params.get("alpha_u", 0.0)),
            target_gc=decode_params.get("target_gc", None),
            target_u=decode_params.get("target_u", None),
            mfe_calc=mfe_calc,
            alpha_mfe=float(decode_params.get("alpha_mfe", 0.0)),
        )

        requests.post(
            f"{server_base_url}/internal/optimizations/{job_id}/complete",
            json={"sequence": mrna},
            timeout=10,
        )

    except Exception as e:
        tb = traceback.format_exc()
        requests.post(
            f"{server_base_url}/internal/optimizations/{job_id}/fail",
            json={"error": str(e), "traceback": tb},
            timeout=10,
        )
        raise
