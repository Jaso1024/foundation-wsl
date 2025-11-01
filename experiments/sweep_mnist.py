#!/usr/bin/env python3
"""
Run many MNIST MLP experiments and record a summary.

Outputs runs/mnist_sweeps/<timestamp>/ with summary.csv and logs.jsonl
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

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from experiments.train_mnist import main as train_main  # type: ignore


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a sweep of MNIST MLP experiments.")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--base-out", type=str, default="runs/mnist")
    p.add_argument("--sweep-out", type=str, default="runs/mnist_sweeps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args(argv)


def sample_cfg(rng: random.Random, epochs: int, batch_size: int, num_workers: int) -> Dict[str, str]:
    lr = rng.choice([5e-4, 1e-3, 3e-3, 1e-2])
    hidden1 = rng.choice([128, 256, 384])
    hidden2 = rng.choice([64, 128, 256])
    seed = rng.randint(1, 10_000_000)
    return {
        "--epochs": str(epochs),
        "--batch-size": str(batch_size),
        "--num-workers": str(num_workers),
        "--lr": str(lr),
        "--hidden1": str(hidden1),
        "--hidden2": str(hidden2),
        "--seed": str(seed),
    }


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    sweep_dir = Path(args.sweep_out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=False)
    summary_csv = sweep_dir / "summary.csv"
    logs_jsonl = sweep_dir / "logs.jsonl"

    fields = [
        "run_index", "seed", "epochs", "lr", "batch_size", "hidden1", "hidden2",
        "run_dir", "train_loss", "train_accuracy", "test_loss", "test_accuracy",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as cf, open(logs_jsonl, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        for i in range(1, args.runs + 1):
            cfg = sample_cfg(rng, args.epochs, args.batch_size, args.num_workers)
            argv_run = [f"--outdir={args.base_out}"] + [f"{k}={v}" for k, v in cfg.items()]
            ret = train_main(argv_run)
            if ret != 0:
                print(f"Run {i} failed with code {ret}", file=sys.stderr)
                continue
            # Read latest run dir
            base_dir = Path(args.base_out)
            latest = max(base_dir.glob("*_seed*"), key=lambda p: p.stat().st_mtime)
            with open(latest / "config.json", "r", encoding="utf-8") as f:
                info = json.load(f)

            row = {
                "run_index": i,
                "seed": info["config"]["seed"],
                "epochs": info["config"]["epochs"],
                "lr": info["config"]["lr"],
                "batch_size": info["config"]["batch_size"],
                "hidden1": info["config"]["hidden1"],
                "hidden2": info["config"]["hidden2"],
                "run_dir": str(latest),
                "train_loss": info["metrics"]["train_loss"],
                "train_accuracy": info["metrics"]["train_accuracy"],
                "test_loss": info["metrics"]["test_loss"],
                "test_accuracy": info["metrics"]["test_accuracy"],
            }
            writer.writerow(row)
            jf.write(json.dumps(row) + "\n")
            cf.flush(); jf.flush()
            if i % 10 == 0:
                print(f"Completed {i}/{args.runs} runs -> {sweep_dir}")
    print(f"Sweep complete -> {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
