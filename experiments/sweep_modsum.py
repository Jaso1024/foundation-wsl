#!/usr/bin/env python3
"""
Run many modsum experiments with varied configs and record a summary.

Outputs a sweep directory: runs/modsum_sweeps/<timestamp>/
- summary.csv: one row per run with config + metrics
- logs.jsonl: per-run JSON entries (config + metrics + run_dir)

Example:
  python experiments/sweep_modsum.py --runs 200
  python experiments/sweep_modsum.py --runs 50 --minN 9 --maxN 25 --base-out runs/modsum
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import the train entrypoint
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from experiments.train_modsum import run_training, parse_args as parse_train_args  # type: ignore


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a sweep of modsum experiments.")
    p.add_argument("--runs", type=int, default=200, help="Number of runs to execute. Default: 200.")
    p.add_argument("--minN", type=int, default=7, help="Min N for sampling (inclusive). Default: 7.")
    p.add_argument("--maxN", type=int, default=31, help="Max N for sampling (inclusive). Default: 31.")
    p.add_argument("--epochs", type=int, default=20, help="Epochs per run. Default: 20.")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size per run. Default: 128.")
    p.add_argument("--base-out", type=str, default="runs/modsum", help="Base output dir for individual runs. Default: runs/modsum")
    p.add_argument("--sweep-out", type=str, default="runs/modsum_sweeps", help="Directory to store sweep summary. Default: runs/modsum_sweeps")
    p.add_argument("--seed", type=int, default=None, help="Sweep seed (controls hyperparam sampling for other hparams). Default: random.")
    p.add_argument("--train-samples", type=int, default=None, help="Training samples per epoch per run; default uses full train pool.")
    p.add_argument("--resample-each-epoch", action="store_true", help="If set with --train-samples, resample a new subset each epoch.")
    return p.parse_args(argv)


def sample_cfg(rng: random.Random, minN: int, maxN: int, epochs: int, batch_size: int, train_samples: int | None, resample_each_epoch: bool) -> Dict[str, str]:
    N = rng.randint(minN, maxN)
    # Keep some variety in other hyperparameters, but epochs and batch_size are fixed
    lr = rng.choice([0.05, 0.1, 0.2, 0.3])
    emb_dim = rng.choice([8, 12, 16, 24])
    hidden_dim = rng.choice([16, 24, 32, 48])
    train_frac = rng.choice([0.7, 0.8, 0.9])
    seed = rng.randint(1, 10_000_000)

    return {
        "--N": str(N),
        "--epochs": str(epochs),
        "--lr": str(lr),
        "--emb-dim": str(emb_dim),
        "--hidden-dim": str(hidden_dim),
        "--batch-size": str(batch_size),
        "--train-frac": str(train_frac),
        "--seed": str(seed),
        **({"--train-samples": str(train_samples)} if train_samples is not None else {}),
        **({"--resample-each-epoch": ""} if resample_each_epoch and train_samples is not None else {}),
    }


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    sweep_dir = Path(args.sweep_out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=False)
    summary_csv = sweep_dir / "summary.csv"
    logs_jsonl = sweep_dir / "logs.jsonl"

    # Prepare CSV
    csv_fields = [
        "run_index",
        "N",
        "epochs",
        "lr",
        "emb_dim",
        "hidden_dim",
        "batch_size",
        "train_frac",
        "seed",
        "run_dir",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as cf, open(logs_jsonl, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()

        for i in range(1, args.runs + 1):
            cfg_args = sample_cfg(rng, args.minN, args.maxN, args.epochs, args.batch_size, args.train_samples, args.resample_each_epoch)
            # Ensure outputs go to the same base dir so train_main creates a new run dir under it
            # train_main uses --outdir to decide base output path
            argv_run = [f"--outdir={args.base_out}"] + [k if v == "" else f"{k}={v}" for k, v in cfg_args.items()]

            # Execute one training run and capture exact run directory
            train_args = parse_train_args(argv_run)
            latest = run_training(train_args)

            # Load metrics from config.json stored by train_main
            with open(latest / "config.json", "r", encoding="utf-8") as f:
                info = json.load(f)

            row = {
                "run_index": i,
                "N": info["config"]["N"],
                "epochs": info["config"]["epochs"],
                "lr": info["config"]["lr"],
                "emb_dim": info["config"]["emb_dim"],
                "hidden_dim": info["config"].get("hidden_dim", None),
                "batch_size": info["config"]["batch_size"],
                "train_frac": info["config"]["train_frac"],
                "seed": info["config"]["seed"],
                "run_dir": str(latest),
                "train_loss": info["metrics"]["train_loss"],
                "train_accuracy": info["metrics"]["train_accuracy"],
                "test_loss": info["metrics"]["test_loss"],
                "test_accuracy": info["metrics"]["test_accuracy"],
            }

            writer.writerow(row)
            jf.write(json.dumps(row) + "\n")
            cf.flush(); jf.flush()

            # Light progress signal
            if i % 10 == 0:
                print(f"Completed {i}/{args.runs} runs -> {sweep_dir}")

    print(f"Sweep complete -> {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
