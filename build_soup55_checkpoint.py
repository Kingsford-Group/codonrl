#!/usr/bin/env python3
"""
Average 55 checkpoint weights and save them as a single .pth file.
The output file contains a state_dict and model config, so later inference only
needs to load this one file.

Output format:
{
    "state_dict": OrderedDict(...),   # averaged model weights
    "config": dict(...),              # model architecture config for build_agent
    "n_checkpoints": 55,              # number of checkpoints included
}

Loading example:
    ckpt = torch.load("soup55.pth", map_location=device)
    agent = build_agent(ckpt["config"], device)
    agent.policy_net.load_state_dict(ckpt["state_dict"])
"""
import torch
import os
import sys
import glob
import argparse
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualizeandbenchmark_multialpha import load_cfg_and_w


def build_soup_checkpoint(checkpoint_dir, output_path, device="cpu"):
    ckpt_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*/ckpt_best_objective.pth")))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    cfg = None
    avg_sd = None
    n = 0

    for ckpt_path in ckpt_paths:
        ckpt_dir_i = os.path.dirname(ckpt_path)
        summary_json = os.path.join(ckpt_dir_i, "training_summary.json")
        if not os.path.exists(summary_json):
            print(f"  Skipped (missing summary): {ckpt_dir_i}")
            continue
        try:
            cfg_i, _ = load_cfg_and_w(summary_json)
            if cfg is None:
                cfg = cfg_i
            sd = torch.load(ckpt_path, map_location=device)
            if avg_sd is None:
                avg_sd = OrderedDict()
                for k, v in sd.items():
                    avg_sd[k] = v.float().clone()
            else:
                for k, v in sd.items():
                    avg_sd[k] += v.float()
            n += 1
            print(f"  [{n}] {os.path.basename(ckpt_dir_i)}")
        except Exception as e:
            print(f"  Skipped {os.path.basename(ckpt_dir_i)}: {e}")
            continue

    if n == 0:
        raise RuntimeError("No valid checkpoints loaded")

    for k in avg_sd:
        avg_sd[k] /= n

    soup = {
        "state_dict": avg_sd,
        "config": cfg,
        "n_checkpoints": n,
    }
    torch.save(soup, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nModel Soup saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  Averaged {n}/{len(ckpt_paths)} checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Model Soup checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./results_linearfold_only",
        # TBD the model checkpoints's folder
    )
    parser.add_argument("--output", type=str, default="soup55.pth")
    args = parser.parse_args()

    build_soup_checkpoint(args.checkpoint_dir, args.output)
