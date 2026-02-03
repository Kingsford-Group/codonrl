

#!/usr/bin/env python3

import torch
import json
import os
import math
from pathlib import Path
import time


def split_protein_sequence(protein_seq, window_size=500, overlap=50):

    fragments = []
    seq_len = len(protein_seq)
    stride = window_size - overlap
    
    for start in range(0, seq_len, stride):
        end = min(start + window_size, seq_len)
        fragment = protein_seq[start:end]
        fragments.append((start, end, fragment))
        
        if end >= seq_len:
            break
    
    return fragments


def calculate_metrics(mrna, w, csc_weights, mfe_calc):

    from visualizeandbenchmark_multialpha import (
        cai_safe, csc_safe, gc_content, u_pct, to_dna
    )
    
    metrics = {}
    
    # CAI
    try:
        metrics['cai'] = cai_safe(mrna, w)
    except Exception as e:
        print(f"    Warning: CAI calculation failed: {e}")
        metrics['cai'] = math.nan
    
    # MFE
    try:
        vienna_mfe = mfe_calc.calculate_vienna_async(mrna).result(timeout=30)
        metrics['vienna_mfe'] = vienna_mfe if vienna_mfe is not None else math.nan
    except Exception as e:
        print(f"    Warning: Vienna MFE calculation failed: {e}")
        metrics['vienna_mfe'] = math.nan
    
    try:
        linearfold_mfe = mfe_calc.calculate_linearfold_async(mrna).result(timeout=30)
        metrics['linearfold_mfe'] = linearfold_mfe if linearfold_mfe is not None else math.nan
    except Exception as e:
        print(f"    Warning: LinearFold MFE calculation failed: {e}")
        metrics['linearfold_mfe'] = math.nan
    
    # CSC
    try:
        metrics['csc'] = csc_safe(mrna, csc_weights)
    except Exception as e:
        print(f"    Warning: CSC calculation failed: {e}")
        metrics['csc'] = math.nan
    
    # GC content
    try:
        metrics['gc_content'] = gc_content(to_dna(mrna))
    except Exception as e:
        print(f"    Warning: GC content calculation failed: {e}")
        metrics['gc_content'] = math.nan
    
    # U content
    try:
        metrics['u_content'] = u_pct(to_dna(mrna))
    except Exception as e:
        print(f"    Warning: U content calculation failed: {e}")
        metrics['u_content'] = math.nan
    
    return metrics





# Update the optimize_long_sequence_ensemble function signature
def optimize_long_sequence_ensemble(
    protein_seq,
    checkpoint_paths,
    window_size=500,
    overlap=50,
    num_checkpoints_per_window=3,
    output_file=None,
    alpha_cai=0.5,
    alpha_csc=0.2,
    alpha_gc=0.1,
    alpha_u=-0.1
):
    """
    Optimize an ultra-long sequence using an ensemble of checkpoints.

    Args:
        protein_seq: Full protein sequence (e.g., 5000 aa).
        checkpoint_paths: List of checkpoint file paths.
        window_size: Window size (amino acids).
        overlap: Overlap size (amino acids).
        num_checkpoints_per_window: Number of checkpoints sampled per window.
        output_file: Output file path (JSON).
        alpha_cai: Weight for CAI.
        alpha_csc: Weight for CSC.
        alpha_gc: Weight for GC content.
        alpha_u: Weight for U content.

    Returns:
        tuple: (full optimized mRNA sequence, metrics dict, performance dict)
    """
    from CodonRL_main import (
        DQNAgent, calculate_relative_adaptiveness,
        HUMAN_FREQ_PER_THOUSAND, AA_TO_CODONS, get_mfe_calculator
    )
    from visualizeandbenchmark_multialpha import (
        load_cfg_and_w, build_agent, multiobjective_decode, load_csc_weights
    )
    
    # Start timing
    start_time = time.time()
    
    # Load CSC weights and MFE calculator (only once)
    csc_file = "./config/csc.json"
    csc_weights = load_csc_weights(csc_file=csc_file)
    mfe_calc = get_mfe_calculator()
    
    # Split the sequence
    fragments = split_protein_sequence(protein_seq, window_size, overlap)
    print(f"Split into {len(fragments)} fragments")
    
    # Store optimization results for each fragment
    optimized_fragments = []
    
    fragment_start_time = time.time()
    
    for i, (start, end, fragment) in enumerate(fragments):
        print(f"\nProcessing fragment {i+1}/{len(fragments)}: position {start}-{end} ({len(fragment)} aa)")
        
        # Randomly select a few checkpoints for this fragment
        import random
        selected_ckpts = random.sample(checkpoint_paths, 
                                      min(num_checkpoints_per_window, len(checkpoint_paths)))
        
        fragment_results = []
        
        for ckpt_path in selected_ckpts:
            try:
                # Get checkpoint directory and summary file
                ckpt_dir = os.path.dirname(ckpt_path)
                summary_json = os.path.join(ckpt_dir, "training_summary.json")
                
                # Check whether the file exists
                if not os.path.exists(summary_json):
                    print(f"  Warning: {summary_json} not found, skipping")
                    continue
                
                # Load configuration (from JSON)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                cfg, w = load_cfg_and_w(summary_json)
                
                # Build the agent
                agent = build_agent(cfg, device)
                
                # Load model weights (from .pth)
                sd = torch.load(ckpt_path, map_location=device)
                agent.policy_net.load_state_dict(sd)
                agent.target_net.load_state_dict(sd)
                agent.policy_net.eval()
                
                # Optimize this fragment using the provided alpha parameters
                mrna_fragment = multiobjective_decode(
                    agent, fragment, w, csc_weights,
                    alpha_cai=alpha_cai,
                    alpha_csc=alpha_csc,
                    alpha_gc=alpha_gc,
                    alpha_u=alpha_u
                )
                
                fragment_results.append(mrna_fragment)
                print(f"  ✓ Succeeded with {os.path.basename(ckpt_dir)}")
                
                # One successful checkpoint is enough (remove this break to ensemble multiple checkpoints)
                break
                
            except Exception as e:
                print(f"  Warning: checkpoint {os.path.basename(ckpt_path)} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not fragment_results:
            raise RuntimeError(f"Fragment {i} failed: all checkpoints failed")
        
        # Ensemble strategy: pick the best result (based on some metric)
        # Simplified: use the first result; you can choose based on CAI/MFE, etc.
        best_mrna = fragment_results[0]
        
        optimized_fragments.append({
            'start': start,
            'end': end,
            'protein': fragment,
            'mrna': best_mrna,
            'overlap_with_next': overlap if end < len(protein_seq) else 0
        })
    
    # Total fragment optimization time
    fragment_optimization_time = time.time() - fragment_start_time
    
    # Merge all fragments
    print("\nMerging fragments...")
    merge_start_time = time.time()
    final_mrna = merge_fragments(optimized_fragments)
    merge_time = time.time() - merge_start_time
    
    # Compute final metrics
    print("\nComputing metrics...")
    metrics_start_time = time.time()
    
    # Need a checkpoint's w for CAI computation
    if checkpoint_paths:
        ckpt_dir = os.path.dirname(checkpoint_paths[0])
        summary_json = os.path.join(ckpt_dir, "training_summary.json")
        if os.path.exists(summary_json):
            _, w = load_cfg_and_w(summary_json)
        else:
            w = calculate_relative_adaptiveness(AA_TO_CODONS, HUMAN_FREQ_PER_THOUSAND)
    else:
        w = calculate_relative_adaptiveness(AA_TO_CODONS, HUMAN_FREQ_PER_THOUSAND)
    
    metrics = calculate_metrics(final_mrna, w, csc_weights, mfe_calc)
    metrics_calculation_time = time.time() - metrics_start_time
    
    # Total time
    total_time = time.time() - start_time
    
    # Performance metrics
    performance = {
        'total_time': total_time,
        'fragment_optimization_time': fragment_optimization_time,
        'merge_time': merge_time,
        'metrics_calculation_time': metrics_calculation_time,
        'time_per_aa': total_time / len(protein_seq) if len(protein_seq) > 0 else 0,
        'time_per_fragment': fragment_optimization_time / len(fragments) if len(fragments) > 0 else 0
    }
    
    # Print metrics
    print("\n" + "="*60)
    print("Optimization complete! Final metrics:")
    print("="*60)
    print(f"  Protein length:     {len(protein_seq)} aa")
    print(f"  mRNA length:        {len(final_mrna)} nt (expected: {len(protein_seq) * 3} nt)")
    print(f"  CAI:           {metrics.get('cai', 'N/A'):.4f}" if not math.isnan(metrics.get('cai', math.nan)) else "  CAI:           N/A")
    print(f"  Vienna MFE:    {metrics.get('vienna_mfe', 'N/A'):.2f} kcal/mol" if not math.isnan(metrics.get('vienna_mfe', math.nan)) else "  Vienna MFE:    N/A")
    print(f"  LinearFold MFE: {metrics.get('linearfold_mfe', 'N/A'):.2f} kcal/mol" if not math.isnan(metrics.get('linearfold_mfe', math.nan)) else "  LinearFold MFE: N/A")
    print(f"  CSC:           {metrics.get('csc', 'N/A'):.4f}" if not math.isnan(metrics.get('csc', math.nan)) else "  CSC:           N/A")
    print(f"  GC content:    {metrics.get('gc_content', 'N/A'):.2%}" if not math.isnan(metrics.get('gc_content', math.nan)) else "  GC content:    N/A")
    print(f"  U content:     {metrics.get('u_content', 'N/A'):.2%}" if not math.isnan(metrics.get('u_content', math.nan)) else "  U content:     N/A")
    print("="*60)
    print("\n[Performance]")
    print(f"  Total runtime:            {performance['total_time']:.2f} s")
    print(f"  Fragment optimization:    {performance['fragment_optimization_time']:.2f} s")
    print(f"  Fragment merging:         {performance['merge_time']:.4f} s")
    print(f"  Metrics calculation:      {performance['metrics_calculation_time']:.2f} s")
    print(f"  Avg time per amino acid:  {performance['time_per_aa']*1000:.2f} ms")
    print(f"  Avg time per fragment:    {performance['time_per_fragment']:.2f} s")
    print("="*60)
    
    # Save results
    if output_file:
        from visualizeandbenchmark_multialpha import to_dna
        result = {
            'protein_sequence': protein_seq,
            'protein_length': len(protein_seq),
            'mrna_sequence_rna': final_mrna,
            'mrna_sequence_dna': to_dna(final_mrna),
            'mrna_length': len(final_mrna),
            'metrics': metrics,
            'performance': performance,
            'optimization_params': {
                'window_size': window_size,
                'overlap': overlap,
                'num_checkpoints_per_window': num_checkpoints_per_window,
                'num_fragments': len(fragments),
                'alpha_cai': alpha_cai,
                'alpha_csc': alpha_csc,
                'alpha_gc': alpha_gc,
                'alpha_u': alpha_u
            }
        }
        
        # Save JSON
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        # Save FASTA
        fasta_rna = output_file.replace('.json', '_rna.fasta')
        fasta_dna = output_file.replace('.json', '_dna.fasta')
        
        with open(fasta_rna, 'w') as f:
            f.write(f">optimized_long_sequence_rna length={len(final_mrna)}\n")
            # 60 characters per line
            for i in range(0, len(final_mrna), 60):
                f.write(final_mrna[i:i+60] + '\n')
        
        with open(fasta_dna, 'w') as f:
            dna_seq = to_dna(final_mrna)
            f.write(f">optimized_long_sequence_dna length={len(dna_seq)}\n")
            for i in range(0, len(dna_seq), 60):
                f.write(dna_seq[i:i+60] + '\n')
        
        print(f"FASTA files saved: {fasta_rna}, {fasta_dna}")
    
    return final_mrna, metrics, performance





def merge_fragments(fragments):
    """
    Merge overlapping mRNA fragments.

    Strategy: drop the trailing overlap from each fragment (except the last),
    then append subsequent fragments from the start (since the previous fragment
    already dropped its overlap).
    """
    if not fragments:
        return ""
    
    if len(fragments) == 1:
        return fragments[0]['mrna']
    
    final_mrna = ""
    
    for i, frag in enumerate(fragments):
        if i == 0:
            # First fragment: drop the trailing overlap
            overlap_codons = frag['overlap_with_next']
            if overlap_codons > 0:
                keep_len = len(frag['mrna']) - (overlap_codons * 3)
                final_mrna += frag['mrna'][:keep_len]
            else:
                final_mrna += frag['mrna']
        else:
            # Subsequent fragments: do not skip, since the previous fragment already dropped its trailing overlap
            # The current fragment's mrna[0] corresponds to its start position in the protein
            overlap_codons = frag['overlap_with_next']
            if overlap_codons > 0 and i < len(fragments) - 1:
                # Not the last fragment: drop the trailing overlap
                keep_len = len(frag['mrna']) - (overlap_codons * 3)
                final_mrna += frag['mrna'][:keep_len]
            else:
                # Last fragment: keep everything
                final_mrna += frag['mrna']
    
    return final_mrna


    

if __name__ == "__main__":
    import glob
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize codon usage for an ultra-long protein sequence")
    parser.add_argument("--protein", type=str, help="Protein sequence")
    parser.add_argument("--protein_file", type=str, help="FASTA file containing the protein sequence")
    parser.add_argument("--window_size", type=int, default=500, help="Window size (amino acids)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap size (amino acids)")
    parser.add_argument("--num_ckpts", type=int, default=3, help="Number of checkpoints per window")
    parser.add_argument("--output", type=str, default="long_sequence_result.json", help="Output JSON file path")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="./checkpoints",
                       help="Checkpoint directory")
    
    # Add alpha parameters
    parser.add_argument("--alpha_cai", type=float, default=2.5, help="CAI weight (default: 2.5)")
    parser.add_argument("--alpha_csc", type=float, default=0, help="CSC weight (default: 0)")
    parser.add_argument("--alpha_gc", type=float, default=0, help="GC content weight (default: 0)")
    parser.add_argument("--alpha_u", type=float, default=1, help="U content weight (default: 1)")
    
    args = parser.parse_args()
    
    # Get the protein sequence
    if args.protein:
        long_protein = args.protein
    elif args.protein_file:
        # Read from a FASTA file
        with open(args.protein_file) as f:
            lines = f.readlines()
            long_protein = ''.join(line.strip() for line in lines if not line.startswith('>'))
    else:
        # Use an example sequence
        print("No protein sequence provided; using an example sequence...")
        long_protein = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK" * 10
    
    # Collect all checkpoint paths
    checkpoint_paths = list(glob.glob(f"{args.checkpoint_dir}/*/ckpt_best_objective.pth"))
    
    print(f"Found {len(checkpoint_paths)} checkpoints")
    print(f"Protein sequence length: {len(long_protein)} aa")
    print("\nOptimization parameters:")
    print(f"  alpha_cai = {args.alpha_cai}")
    print(f"  alpha_csc = {args.alpha_csc}")
    print(f"  alpha_gc  = {args.alpha_gc}")
    print(f"  alpha_u   = {args.alpha_u}")
    
    # Optimize
    optimized_mrna, metrics, performance = optimize_long_sequence_ensemble(
        long_protein,
        checkpoint_paths,
        window_size=args.window_size,
        overlap=args.overlap,
        num_checkpoints_per_window=args.num_ckpts,
        output_file=args.output,
        alpha_cai=args.alpha_cai,
        alpha_csc=args.alpha_csc,
        alpha_gc=args.alpha_gc,
        alpha_u=args.alpha_u
    )
    
    
    
# # 1. Use the default example sequence
# python long_sequence_inference.py

# # 2. Provide a protein sequence
# python long_sequence_inference.py --protein "MSKGEELFTGVV..."

# # 3. Read from a FASTA file
# python long_sequence_inference.py --protein_file my_protein.fasta --output my_result.json

# # 4. Customize parameters
# python long_sequence_inference.py \
#     --protein_file input.fasta \
#     --window_size 400 \
#     --overlap 100 \
#     --num_ckpts 5 \
#     --output results/optimized.json \
#     --checkpoint_dir /path/to/results_linearfold_only



# python long_sequence_inference.py --protein_file ./GEMORNA/Q3L8U1.fasta --output CodonRL_Q3L8U1.json



# # Use default parameters
# python long_sequence_inference.py --protein_file ./Q3L8U1.fasta --output CodonRL_Q3L8U1_setting2.json

# # Customize alpha parameters
# python long_sequence_inference.py \
#     --protein_file ./Q3L8U1.fasta \
#     --output CodonRL_Q3L8U1.json \
#     --alpha_cai 0.6 \
#     --alpha_csc 0.3 \
#     --alpha_gc 0.05 \
#     --alpha_u -0.15

# # Optimize MFE (increase negative U weight)
# python long_sequence_inference.py \
#     --protein_file my_protein.fasta \
#     --output result.json \
#     --alpha_cai 0.4 \
#     --alpha_csc 0.3 \
#     --alpha_gc 0.1 \
#     --alpha_u -0.3

# # Show help
# python long_sequence_inference.py --help
