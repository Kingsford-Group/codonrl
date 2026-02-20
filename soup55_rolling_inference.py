#!/usr/bin/env python3
"""
Model Soup + Rolling Window inference (step=1).
Loads pre-built soup55.pth, inference speed = single model.
"""
import torch
import json
import os
import sys
import math
import time
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CodonRL_main import (
    CodonRL, calculate_relative_adaptiveness,
    HUMAN_FREQ_PER_THOUSAND, ECOLLI_K12_FREQ_PER_THOUSAND,
    AA_TO_CODONS, CODON_TO_INT, INT_TO_CODON,
    configure_target_w_table, get_mfe_calculator
)
from visualizeandbenchmark_multiobjective_multialpha import (
    build_agent, load_csc_weights,
    cai_safe, csc_safe, gc_content, u_pct, to_dna, to_rna,
    calculate_codon_gc, calculate_codon_u
)


def load_soup_checkpoint(soup_path, device):
    ckpt = torch.load(soup_path, map_location=device)
    cfg = ckpt["config"]

    table = cfg.get("codon_table", "human").lower()
    if table == "human":
        freq = HUMAN_FREQ_PER_THOUSAND
    elif table in ("ecolik12", "ecoli", "ecolik-12", "e.coli"):
        freq = ECOLLI_K12_FREQ_PER_THOUSAND
    else:
        raise ValueError(f"Unsupported codon table: {table}")
    w = calculate_relative_adaptiveness(AA_TO_CODONS, freq)
    configure_target_w_table(w)

    agent = build_agent(cfg, device)
    agent.policy_net.load_state_dict(ckpt["state_dict"])
    agent.policy_net.eval()

    n = ckpt.get("n_checkpoints", "?")
    print(f"Soup checkpoint loaded: {n} checkpoints averaged")
    return agent, w, cfg


def rolling_decode(agent, protein, w, csc_weights, window_size=500,
                   alpha_cai=0.5, alpha_csc=0.0, alpha_gc=0.0, alpha_u=0.0):
    logw = {c: math.log(max(w.get(c, 1e-12), 1e-12)) for c in w}
    logcsc = {c: math.log(max(csc_weights.get(c, 1e-12), 1e-12)) for c in csc_weights}

    total_len = len(protein)
    mrna_codons = []
    mrna = ""

    cached_window_start = None

    for pos in range(total_len):
        if pos < window_size:
            win_start = 0
            win_protein = protein[0:pos + 1]
            mrna_context = mrna
            pos_in_window = pos
        else:
            win_start = pos - window_size + 1
            win_protein = protein[win_start:pos + 1]
            mrna_context = "".join(mrna_codons[win_start:pos])
            pos_in_window = window_size - 1

        if cached_window_start != win_start:
            agent._precompute_protein_memory(win_protein)
            cached_window_start = win_start

        state = agent._get_state(mrna_context, pos_in_window)

        with torch.no_grad():
            q = agent.policy_net.decode_mrna(
                state["mrna"], state["pos"],
                agent.protein_memory_cache, agent.protein_pad_mask_cache
            )[0]

        aa = protein[pos]
        candidate_codons = AA_TO_CODONS[aa]
        best_score = -float('inf')
        best_codon = candidate_codons[0]

        for codon in candidate_codons:
            idx = CODON_TO_INT[codon]
            score = q[idx].item()
            if alpha_cai != 0:
                score += alpha_cai * logw.get(codon, 0)
            if alpha_csc != 0:
                score += alpha_csc * logcsc.get(codon, 0)
            if alpha_gc != 0:
                score += alpha_gc * calculate_codon_gc(codon)
            if alpha_u != 0:
                score += alpha_u * (-calculate_codon_u(codon))
            if score > best_score:
                best_score = score
                best_codon = codon

        mrna_codons.append(best_codon)
        mrna += best_codon

        if (pos + 1) % 200 == 0 or pos + 1 == total_len:
            print(f"  decoded {pos+1}/{total_len} aa")

    return mrna


def calc_metrics(mrna, w, csc_weights, mfe_calc, label=""):
    m = {}
    try: m['cai'] = cai_safe(mrna, w)
    except: m['cai'] = math.nan
    try: m['csc'] = csc_safe(mrna, csc_weights)
    except: m['csc'] = math.nan
    try: m['gc'] = gc_content(to_dna(mrna))
    except: m['gc'] = math.nan
    try: m['u_pct'] = u_pct(to_dna(mrna))
    except: m['u_pct'] = math.nan
    try: m['vienna_mfe'] = mfe_calc.calculate_vienna_async(mrna).result(timeout=300)
    except: m['vienna_mfe'] = math.nan
    try: m['linearfold_mfe'] = mfe_calc.calculate_linearfold_async(mrna).result(timeout=120)
    except: m['linearfold_mfe'] = math.nan

    if label:
        print(f"\n  [{label}]")
        print(f"    length:        {len(mrna)} nt")
        for k, v in m.items():
            if isinstance(v, float) and not math.isnan(v):
                print(f"    {k:15s}: {v:.4f}")
            else:
                print(f"    {k:15s}: N/A")
    return m


def load_fasta_seq(fasta_path, as_rna=False):
    with open(fasta_path) as f:
        lines = f.readlines()
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    return to_rna(seq) if as_rna else seq


def print_comparison(ours, baseline):
    print("\n" + "=" * 65)
    print(f"{'Metric':>18s} {'Soup55-Rolling':>18s} {'GEMORNA':>18s}")
    print("-" * 65)
    for k in ours:
        v1 = ours.get(k, math.nan)
        v2 = baseline.get(k, math.nan)
        s1 = f"{v1:.4f}" if isinstance(v1, (int, float)) and not math.isnan(v1) else "N/A"
        s2 = f"{v2:.4f}" if isinstance(v2, (int, float)) and not math.isnan(v2) else "N/A"
        print(f"{k:>18s} {s1:>18s} {s2:>18s}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Model Soup rolling window inference (step=1)")
    parser.add_argument("--protein_file", type=str,
                        default="./Q3L8U1.fasta")
    parser.add_argument("--gemorna_file", type=str,
                        default="./Q3L8U1_gemorna.fasta")
    parser.add_argument("--soup_checkpoint", type=str,
                        default="./model_soup_checkpoint/soup55.pth")
    parser.add_argument("--output", type=str, default="soup55_rolling_Q3L8U1.json")
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--alpha_cai", type=float, default=2.5)
    parser.add_argument("--alpha_csc", type=float, default=0)
    parser.add_argument("--alpha_gc", type=float, default=0)
    parser.add_argument("--alpha_u", type=float, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    protein = load_fasta_seq(args.protein_file)
    print(f"Protein: {len(protein)} aa")

    print(f"\nLoading {args.soup_checkpoint} ...")
    t0 = time.time()
    agent, w, cfg = load_soup_checkpoint(args.soup_checkpoint, device)
    load_time = time.time() - t0
    print(f"Load time: {load_time:.2f}s")

    csc_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csc.json")
    csc_weights = load_csc_weights(csc_file=csc_file)
    mfe_calc = get_mfe_calculator()

    print(f"\nRolling decode: window_size={args.window_size}, step=1")
    print(f"alpha_cai={args.alpha_cai}, alpha_csc={args.alpha_csc}, "
          f"alpha_gc={args.alpha_gc}, alpha_u={args.alpha_u}")

    t0 = time.time()
    our_mrna = rolling_decode(
        agent, protein, w, csc_weights,
        window_size=args.window_size,
        alpha_cai=args.alpha_cai, alpha_csc=args.alpha_csc,
        alpha_gc=args.alpha_gc, alpha_u=args.alpha_u
    )
    infer_time = time.time() - t0
    print(f"\nTotal inference time: {infer_time:.1f}s ({infer_time/len(protein)*1000:.1f} ms/aa)")

    print("\nComputing Soup55-Rolling metrics...")
    our_metrics = calc_metrics(our_mrna, w, csc_weights, mfe_calc, label="Soup55-Rolling")

    gemorna_metrics = {}
    if args.gemorna_file and os.path.exists(args.gemorna_file):
        print("\nComputing GEMORNA baseline metrics...")
        gemorna_mrna = load_fasta_seq(args.gemorna_file, as_rna=True)
        gemorna_metrics = calc_metrics(gemorna_mrna, w, csc_weights, mfe_calc, label="GEMORNA")
        print_comparison(our_metrics, gemorna_metrics)

    result = {
        'protein_length': len(protein),
        'mrna_length': len(our_mrna),
        'mrna_rna': our_mrna,
        'mrna_dna': to_dna(our_mrna),
        'metrics': our_metrics,
        'gemorna_metrics': gemorna_metrics,
        'params': {
            'method': 'model_soup_rolling',
            'soup_checkpoint': args.soup_checkpoint,
            'window_size': args.window_size,
            'step': 1,
            'alpha_cai': args.alpha_cai,
            'alpha_csc': args.alpha_csc,
            'alpha_gc': args.alpha_gc,
            'alpha_u': args.alpha_u,
        },
        'timing': {
            'checkpoint_load_s': load_time,
            'inference_s': infer_time,
            'ms_per_aa': infer_time / len(protein) * 1000,
        }
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print(f"\nResults saved: {args.output}")

    fasta_out = args.output.replace('.json', '_rna.fasta')
    with open(fasta_out, 'w') as f:
        f.write(f">CodonRL_soup55_rolling length={len(our_mrna)}\n")
        for i in range(0, len(our_mrna), 60):
            f.write(our_mrna[i:i+60] + '\n')
    print(f"FASTA saved: {fasta_out}")


if __name__ == "__main__":
    main()