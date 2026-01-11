import os, csv, json, math, argparse, sys
sys.path.append('/home/shiyid/codebase/rl_mrndesign')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from CodonRL_short_multigpu_withrecords_linearfoldablation import (
    DQNAgent, configure_target_w_table, calculate_relative_adaptiveness,
    HUMAN_FREQ_PER_THOUSAND, ECOLLI_K12_FREQ_PER_THOUSAND,
    AA_TO_CODONS, CODON_TO_INT, INT_TO_CODON, calculate_cai,
    get_mfe_calculator
)

# ============================================================================
# CSC (Codon Stability Coefficient) data
# ============================================================================
# CSC should be based on mRNA half-life data, independent of CAI
# Reference: Wu et al. 2019, "Translation affects mRNA stability in a codon-dependent manner in human cells"
# 
# Two approaches are provided:
# 1. If real CSC data file exists, load from file
# 2. If not, use a simplified model (GC content-based approximation)

CSC_WEIGHTS = None  # Global variable to store CSC weights

def load_csc_weights(csc_file=None, codon_table="human"):
    """
    Load CSC weight data
    
    Args:
        csc_file: CSC data file path (JSON format, keys are codons, values are stability coefficients)
        codon_table: Which codon table to use for generating approximations if no file is provided
    
    Returns:
        dict: Codon -> CSC value dictionary
    """
    global CSC_WEIGHTS
    
    if csc_file and os.path.exists(csc_file):
        # Load real CSC data from file
        with open(csc_file) as f:
            CSC_WEIGHTS = json.load(f)
        print(f"Loaded CSC weights from {csc_file}")
        return CSC_WEIGHTS
    
    # If no real data available, use a simplified model
    # Based on observation: GC-rich codons typically correspond to more stable mRNA
    # This is only a rough approximation and should be replaced with real data
    print("WARNING: Using simplified CSC model based on GC content.")
    print("         For accurate results, provide real CSC data from mRNA stability experiments.")
    
    CSC_WEIGHTS = {}
    for aa, codons in AA_TO_CODONS.items():
        for codon in codons:
            # Calculate codon GC content
            gc_count = codon.count('G') + codon.count('C')
            # Simplified model: CSC centered around 1.0, range ~0.5-1.5
            # Higher GC content codons have slightly higher CSC
            CSC_WEIGHTS[codon] = 0.7 + 0.6 * (gc_count / 3.0)
            # Add some randomness to simulate differences between codons
            # (In real data, synonymous codons can have significant CSC differences)
            CSC_WEIGHTS[codon] *= (0.9 + 0.2 * hash(codon) % 100 / 100.0)
    
    return CSC_WEIGHTS

def calculate_csc(mrna: str, csc_weights: dict) -> float:
    """
    Calculate CSC (Codon Stability Coefficient)
    
    Args:
        mrna: mRNA sequence
        csc_weights: Codon stability coefficient dictionary
    
    Returns:
        float: CSC value (geometric mean)
    """
    codons = split_codons(mrna)
    if not codons:
        return math.nan
    
    vals = []
    for c in codons:
        v = csc_weights.get(c, None)
        if v is None:
            return math.nan
        vals.append(max(v, 1e-12))
    
    # # Use geometric mean (similar calculation to CAI, but completely different data source)
    # logs = [math.log(v) for v in vals]
    # return math.exp(sum(logs) / len(logs)) if logs else math.nan
    return sum(vals) / len(vals) if vals else math.nan

# ============================================================================
# Helper functions
# ============================================================================

def to_float(x):
    if x is None: return math.nan
    s = str(x).strip()
    if s == "" or s.lower() == "na": return math.nan
    try: return float(s)
    except: return math.nan

def to_dna(seq: str) -> str:
    return (seq or "").strip().upper().replace('U', 'T')

def to_rna(seq: str) -> str:
    return (seq or "").strip().upper().replace('T', 'U')

def split_codons(s: str):
    s = (s or "").strip().upper()
    return [s[i:i+3] for i in range(0, len(s), 3)] if len(s) % 3 == 0 else []

def load_cfg_and_w(summary_json_path):
    with open(summary_json_path) as f:
        summ = json.load(f)
    cfg = summ["config"]
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

def build_agent(cfg, device):
    cfg = dict(cfg)
    cfg["device"] = device
    cfg["use_amp"] = False
    cfg["eps_start"] = 0.0
    cfg["eps_end"] = 0.0
    cfg["eps_decay"] = 1
    return DQNAgent(cfg)

def hybrid_decode(agent, protein, w, alpha=0.5):
    logw = {c: (math.log(max(w.get(c, 1e-12), 1e-12))) for c in w}
    agent._precompute_protein_memory(protein)
    mrna = ""
    for t, aa in enumerate(protein):
        state = agent._get_state(mrna, t)
        with torch.no_grad():
            q = agent.policy_net.decode_mrna(
                state["mrna"], state["pos"],
                agent.protein_memory_cache, agent.protein_pad_mask_cache
            )[0]
            idxs = [CODON_TO_INT[c] for c in AA_TO_CODONS[aa]]
            scores = [(q[i].item() + alpha * logw[INT_TO_CODON[i]], i) for i in idxs]
            _, i_best = max(scores, key=lambda x: x[0])
        mrna += INT_TO_CODON[i_best]
    return mrna

def choose_string_for_w(mrna: str, w: dict) -> str:
    codons = split_codons(mrna)
    if codons and all(c in w for c in codons):
        return mrna
    mrna_rna = to_rna(mrna)
    codons_rna = split_codons(mrna_rna)
    if codons_rna and all(c in w for c in codons_rna):
        return mrna_rna
    mrna_dna = to_dna(mrna)
    codons_dna = split_codons(mrna_dna)
    if codons_dna and all(c in w for c in codons_dna):
        return mrna_dna
    return mrna

def cai_safe(mrna: str, w: dict) -> float:
    s = choose_string_for_w(mrna, w)
    return calculate_cai(s, w)

def csc_safe(mrna: str, csc_weights: dict) -> float:
    """
    Safely calculate CSC, handling DNA/RNA format conversion
    
    Note: CSC uses independent csc_weights, completely different from CAI's w dictionary
    """
    s = choose_string_for_w(mrna, csc_weights)
    return calculate_csc(s, csc_weights)

def gc_content(seq: str) -> float:
    s = (seq or "").strip().upper()
    if not s: return math.nan
    return (s.count('G') + s.count('C')) / len(s)

def u_pct(seq: str) -> float:
    s = (seq or "").strip().upper()
    if not s: return math.nan
    return (s.count('U') + s.count('T')) / len(s)

def run_benchmark(csv_path, ckpt_root, alpha, out_csv,
                  w_cai=1.0, w_mfe=1.0, w_csc=0.0, w_gc=0.0, w_u=0.0,
                  csc_file=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mfe_calc = get_mfe_calculator()
    
    # Load CSC weights (independent of CAI)
    csc_weights = load_csc_weights(csc_file)

    out_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                idx = str(int(r["index"]))
            except:
                continue
            protein = r["protein_seq"].strip().upper()
            if not protein or any(aa not in AA_TO_CODONS for aa in protein):
                continue

            ckpt_dir = os.path.join(ckpt_root, f"{idx}_linearfold_linearfold")
            summary_json = os.path.join(ckpt_dir, "training_summary.json")
            ckpt_path = os.path.join(ckpt_dir, "ckpt_best_objective.pth")
            if not (os.path.exists(summary_json) and os.path.exists(ckpt_path)):
                continue

            cfg, w = load_cfg_and_w(summary_json)  # w for CAI
            agent = build_agent(cfg, device)
            sd = torch.load(ckpt_path, map_location=device)
            agent.policy_net.load_state_dict(sd)
            agent.target_net.load_state_dict(sd)

            mrna = hybrid_decode(agent, protein, w, alpha=alpha)

            # ours metrics
            cai_ours = cai_safe(mrna, w)  # Use CAI's w dictionary
            try:
                vienna_mfe = mfe_calc.calculate_vienna_async(mrna).result()
            except Exception:
                vienna_mfe = None
            try:
                linearfold_mfe = mfe_calc.calculate_linearfold_async(mrna).result()
            except Exception:
                linearfold_mfe = None
            mfe_ours = vienna_mfe if (vienna_mfe is not None) else linearfold_mfe

            csc_ours = csc_safe(mrna, csc_weights)  # Use independent CSC weights
            gc_ours = gc_content(to_dna(mrna))
            u_ours  = u_pct(to_dna(mrna))

            # base metrics from CSV (compatible with two column name formats)
            cai_base = to_float(r.get("cai_gemorna") or r.get("cai"))
            mfe_base = to_float(r.get("mfe_gemorna") or r.get("mfe"))
            csc_base = to_float(r.get("csc_gemorna"))
            gc_base  = to_float(r.get("gc_gemorna"))
            u_base   = to_float(r.get("u_pct_gemorna"))

            # baseline CDS
            cds_gemorna = r.get("cds_seq", "").strip()

            # deltas
            delta_cai = (cai_ours - cai_base) if (np.isfinite(cai_base)) else math.nan
            delta_mfe = (mfe_ours - mfe_base) if (mfe_ours is not None and np.isfinite(mfe_base)) else math.nan
            delta_csc = (csc_ours - csc_base) if (np.isfinite(csc_base)) else math.nan
            delta_gc  = (gc_ours - gc_base) if (np.isfinite(gc_base)) else math.nan
            delta_u   = (u_ours - u_base) if (np.isfinite(u_base)) else math.nan

            # weighted composite score
            score = 0.0
            if np.isfinite(delta_cai): score += w_cai * delta_cai
            if np.isfinite(delta_mfe): score += (-w_mfe) * delta_mfe
            if np.isfinite(delta_csc): score += w_csc * delta_csc
            if np.isfinite(delta_gc):  score += w_gc  * delta_gc
            if np.isfinite(delta_u):   score += w_u   * delta_u

            out_rows.append({
                "index": idx,
                # base
                "cai_gemorna": cai_base if np.isfinite(cai_base) else "",
                "mfe_gemorna": mfe_base if np.isfinite(mfe_base) else "",
                "csc_gemorna": csc_base if np.isfinite(csc_base) else "",
                "gc_gemorna":  gc_base  if np.isfinite(gc_base)  else "",
                "u_pct_gemorna": u_base if np.isfinite(u_base) else "",
                # ours
                "cai_ours": cai_ours,
                "vienna_mfe_ours": vienna_mfe if vienna_mfe is not None else "",
                "linearfold_mfe_ours": linearfold_mfe if linearfold_mfe is not None else "",
                "mfe_ours": mfe_ours if mfe_ours is not None else "",
                "csc_ours": csc_ours if np.isfinite(csc_ours) else "",
                "gc_ours":  gc_ours  if np.isfinite(gc_ours)  else "",
                "u_pct_ours": u_ours if np.isfinite(u_ours) else "",
                # deltas
                "delta_cai": delta_cai if np.isfinite(delta_cai) else "",
                "delta_vienna_mfe": ((vienna_mfe - mfe_base) if (vienna_mfe is not None and np.isfinite(mfe_base)) else ""),
                "delta_mfe": delta_mfe if np.isfinite(delta_mfe) else "",
                "delta_csc": delta_csc if np.isfinite(delta_csc) else "",
                "delta_gc":  delta_gc  if np.isfinite(delta_gc)  else "",
                "delta_u":   delta_u   if np.isfinite(delta_u)   else "",
                # composite
                "score": score,
                # sequences
                "cds_gemorna": cds_gemorna,
                "cds_ours_rna": mrna,
                "cds_ours_dna": to_dna(mrna),
                # parameters
                "alpha": alpha,
                "w_cai": w_cai,
                "w_mfe": w_mfe,
                "w_csc": w_csc,
                "w_gc": w_gc,
                "w_u": w_u,
            })
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "index",
            "cai_gemorna","mfe_gemorna","csc_gemorna","gc_gemorna","u_pct_gemorna",
            "cai_ours","vienna_mfe_ours","linearfold_mfe_ours","mfe_ours","csc_ours","gc_ours","u_pct_ours",
            "delta_cai","delta_vienna_mfe","delta_mfe","delta_csc","delta_gc","delta_u",
            "score",
            "cds_gemorna","cds_ours_rna","cds_ours_dna",
            "alpha","w_cai","w_mfe","w_csc","w_gc","w_u",
        ]
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for row in out_rows:
            wcsv.writerow(row)
    
    # Export FASTA files
    base = os.path.splitext(os.path.basename(out_csv))[0]
    fa_rna = os.path.join(os.path.dirname(out_csv), f"{base}_ours_rna.fasta")
    fa_dna = os.path.join(os.path.dirname(out_csv), f"{base}_ours_dna.fasta")
    with open(fa_rna, "w") as fr, open(fa_dna, "w") as fd:
        for row in out_rows:
            idx = row["index"]
            rna = row.get("cds_ours_rna", "")
            dna = row.get("cds_ours_dna", "")
            if rna:
                fr.write(f">{idx}\n{rna}\n")
            if dna:
                fd.write(f">{idx}\n{dna}\n")
    print(f"Saved FASTA -> {fa_rna}")
    print(f"Saved FASTA -> {fa_dna}")
    
    return out_rows

def visualize(rows, csv_path, outdir, title=None, txt_path=None):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_png = os.path.join(outdir, f"{base}_viz.png")
    os.makedirs(outdir, exist_ok=True)

    def arr(name):
        return np.array([to_float(r.get(name)) for r in rows], dtype=float)

    # baseline vs ours
    cai_g, cai_o = arr("cai_gemorna"), arr("cai_ours")
    mfe_g, mfe_o_v, mfe_o_lf = arr("mfe_gemorna"), arr("vienna_mfe_ours"), arr("linearfold_mfe_ours")
    mfe_o = np.where(np.isfinite(mfe_o_v), mfe_o_v, mfe_o_lf)

    csc_g, csc_o = arr("csc_gemorna"), arr("csc_ours")
    gc_g,  gc_o  = arr("gc_gemorna"),  arr("gc_ours")
    u_g,   u_o   = arr("u_pct_gemorna"), arr("u_pct_ours")

    score = arr("score")

    valid_cai = np.isfinite(cai_g) & np.isfinite(cai_o)
    valid_mfe = np.isfinite(mfe_g) & np.isfinite(mfe_o)
    valid_csc = np.isfinite(csc_g) & np.isfinite(csc_o)
    valid_gc  = np.isfinite(gc_g)  & np.isfinite(gc_o)
    valid_u   = np.isfinite(u_g)   & np.isfinite(u_o)

    delta_cai = np.where(valid_cai, cai_o - cai_g, np.nan)
    delta_mfe = np.where(valid_mfe, mfe_o - mfe_g, np.nan)
    delta_csc = np.where(valid_csc, csc_o - csc_g, np.nan)
    delta_gc  = np.where(valid_gc,  gc_o  - gc_g,  np.nan)
    delta_u   = np.where(valid_u,   u_o   - u_g,   np.nan)

    better_cai = np.where(valid_cai, cai_o > cai_g, False)
    better_mfe = np.where(valid_mfe, mfe_o < mfe_g, False)
    better_csc = np.where(valid_csc, csc_o > csc_g, False)
    better_gc  = np.where(valid_gc,  gc_o  > gc_g,  False)
    better_u   = np.where(valid_u,   u_o   > u_g,   False)

    n_total = len(rows)
    summary = [
        f"Total rows: {n_total}",
        f"CAI better:   {int(np.nansum(better_cai))} / {int(np.nansum(valid_cai))}",
        f"MFE better:   {int(np.nansum(better_mfe))} / {int(np.nansum(valid_mfe))}  (more negative is better)",
        f"CSC better:   {int(np.nansum(better_csc))} / {int(np.nansum(valid_csc))}",
        f"GC better:    {int(np.nansum(better_gc))} / {int(np.nansum(valid_gc))}",
        f"U% better:    {int(np.nansum(better_u))} / {int(np.nansum(valid_u))}",
        f"Score > 0:    {int(np.nansum(score > 0)) if np.any(np.isfinite(score)) else 0}",
    ]
    print("\n".join(summary))
    if txt_path:
        with open(txt_path, "w") as f:
            f.write(os.path.relpath(csv_path, start=os.path.dirname(txt_path)) + " \n")
            for s in summary:
                f.write(s + "\n")

    # 2x3: 5 parity plots + 1 score histogram
    plt.figure(figsize=(16, 10))
    plt.suptitle(title or f"Benchmark vs GeMoRNA ({base})", fontsize=14)

    def parity(ax, gx, ox, valid, xlabel, ylabel, better_rule):
        x, y = gx[valid], ox[valid]
        if len(x) == 0:
            ax.set_title(f"{xlabel} vs {ylabel} (no valid)")
            ax.axis("off")
            return
        colors = np.where(better_rule(gx, ox, valid), "#2ca02c", "#d62728")
        ax.scatter(x, y, c=colors, s=20, alpha=0.7, edgecolors="none")
        mn, mx = min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)

    ax1 = plt.subplot(2, 3, 1)
    parity(ax1, cai_g, cai_o, valid_cai, "GeMoRNA CAI", "Ours CAI",
           lambda gx, ox, valid: (ox[valid] > gx[valid]))

    ax2 = plt.subplot(2, 3, 2)
    parity(ax2, mfe_g, mfe_o, valid_mfe, "GeMoRNA MFE", "Ours MFE (Vienna preferred)",
           lambda gx, ox, valid: (ox[valid] < gx[valid]))

    ax3 = plt.subplot(2, 3, 3)
    parity(ax3, csc_g, csc_o, valid_csc, "GeMoRNA CSC", "Ours CSC",
           lambda gx, ox, valid: (ox[valid] > gx[valid]))

    ax4 = plt.subplot(2, 3, 4)
    parity(ax4, gc_g, gc_o, valid_gc, "GeMoRNA GC (0-1)", "Ours GC (0-1)",
           lambda gx, ox, valid: (ox[valid] > gx[valid]))

    ax5 = plt.subplot(2, 3, 5)
    parity(ax5, u_g, u_o, valid_u, "GeMoRNA U/T (0-1)", "Ours U/T (0-1)",
           lambda gx, ox, valid: (ox[valid] > gx[valid]))

    ax6 = plt.subplot(2, 3, 6)
    s = score[np.isfinite(score)]
    if len(s):
        ax6.hist(s, bins=30, color="#9467bd", alpha=0.85)
        ax6.axvline(0, color="k", linestyle="--", linewidth=1)
        ax6.set_xlabel("Composite score"); ax6.set_ylabel("Count"); ax6.grid(alpha=0.3)
    else:
        ax6.set_title("Composite score (no data)"); ax6.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot -> {out_png}")

    # Delta histograms
    out_png_delta = os.path.join(outdir, f"{base}_delta_viz.png")
    plt.figure(figsize=(12, 9))
    plt.suptitle("Delta distributions (ours - GeMoRNA)", fontsize=14)

    def hist_delta(ax, d, label, color):
        d = d[np.isfinite(d)]
        if len(d): ax.hist(d, bins=30, color=color, alpha=0.85)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel(label); ax.set_ylabel("Count"); ax.grid(alpha=0.3)

    axd1 = plt.subplot(2, 2, 1); hist_delta(axd1, delta_cai, "ΔCAI", "#1f77b4")
    axd2 = plt.subplot(2, 2, 2); hist_delta(axd2, delta_mfe, "ΔMFE (more negative is better)", "#ff7f0e")
    axd3 = plt.subplot(2, 2, 3); hist_delta(axd3, delta_csc, "ΔCSC", "#2ca02c")
    axd4 = plt.subplot(2, 2, 4); hist_delta(axd4, delta_gc,  "ΔGC",  "#d62728")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png_delta, dpi=150)
    print(f"Saved plot -> {out_png_delta}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="gemorna_with_cai.csv or gemorna_with_all_metrics.csv path")
    ap.add_argument("--ckpt_root", required=True, help="results_linearfold_only root")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run_name", default=None, help="prefix for outputs (e.g., alpha0.5)")
    ap.add_argument("--title", default=None)
    # weights for composite score
    ap.add_argument("--w_cai", type=float, default=1.0)
    ap.add_argument("--w_mfe", type=float, default=1.0)
    ap.add_argument("--w_csc", type=float, default=0.0)
    ap.add_argument("--w_gc",  type=float, default=0.0)
    ap.add_argument("--w_u",   type=float, default=0.0)
    # CSC data file
    ap.add_argument("--csc_file", default=None, 
                    help="JSON file with CSC weights (codon -> stability coefficient)")
    args = ap.parse_args()

    # Automatically generate subdirectory name based on parameters
    if args.run_name:
        subdir_name = args.run_name
    else:
        subdir_name = f"alpha{args.alpha}_cai{args.w_cai}_mfe{args.w_mfe}_csc{args.w_csc}_gc{args.w_gc}_u{args.w_u}"
    
    # Create subdirectory under outdir
    output_dir = os.path.join(args.outdir, subdir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = args.run_name or f"gemorna_hybrid_alpha{args.alpha}"
    out_csv = os.path.join(output_dir, f"{base_name}.csv")
    out_txt = os.path.join(output_dir, f"{args.alpha}.txt")

    rows = run_benchmark(
        args.csv, args.ckpt_root, args.alpha, out_csv,
        w_cai=args.w_cai, w_mfe=args.w_mfe, w_csc=args.w_csc, 
        w_gc=args.w_gc, w_u=args.w_u,
        csc_file=args.csc_file
    )
    visualize(rows, out_csv, output_dir, title=args.title or f"Hybrid α={args.alpha} vs GeMoRNA", txt_path=out_txt)

if __name__ == "__main__":
    main()

# Example usage:
# 
# 1. Using simplified CSC model (GC content-based approximation):
# python visualizeandbenchmark_multiobjective_fixed.py \
#   --csv /path/to/gemorna_with_all_metrics.csv \
#   --ckpt_root /path/to/results_linearfold_only \
#   --alpha 0.5 \
#   --w_cai 1.0 --w_mfe 0.05 --w_csc 0.30 --w_gc 0.10 --w_u 0.05 \
#   --outdir /path/to/benchmark_outputs \
#   --run_name gemorna_multi_balanced
#
# 2. Using real CSC data file:
# python visualizeandbenchmark_multiobjective_fixed.py \
#   --csv /path/to/gemorna_with_all_metrics.csv \
#   --ckpt_root /path/to/results_linearfold_only \
#   --alpha 0.5 \
#   --w_cai 1.0 --w_mfe 0.05 --w_csc 0.30 --w_gc 0.10 --w_u 0.05 \
#   --csc_file /path/to/csc_weights.json \
#   --outdir /path/to/benchmark_outputs \
#   --run_name gemorna_multi_balanced