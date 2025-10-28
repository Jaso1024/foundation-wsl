#!/usr/bin/env python3
"""
Evaluate the weights->metadata Transformer on a holdout set and
write denormalized predictions alongside ground truth and MAE per field.

Outputs under runs/weights2meta/eval-<timestamp>/:
- predictions.csv (one row per run)
- mae.json (per-field MAE)
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from experiments.weights2meta_train import (  # type: ignore
    TARGET_FIELDS,
    list_run_dirs,
    load_example,
    WeightsTransformer,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate weights2meta model.")
    ap.add_argument("--runs-base", type=str, default="runs/modsum")
    ap.add_argument("--model", type=str, default="runs/weights2meta/weights2meta.pt")
    ap.add_argument("--stats", type=str, default="runs/weights2meta/y_norm.json")
    ap.add_argument("--outdir", type=str, default="runs/weights2meta")
    ap.add_argument("--holdout-frac", type=float, default=0.1, help="Fraction of runs for holdout eval.")
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()


def denorm(y: np.ndarray, stats: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    vals = {}
    for i, k in enumerate(TARGET_FIELDS):
        mean, std = stats[k]
        vals[k] = float(y[i] * std + mean)
    return vals


def main() -> int:
    args = parse_args()
    base = Path(args.runs_base)
    run_dirs = list_run_dirs(base)
    if len(run_dirs) < 10:
        print(f"Not enough runs under {base} (found {len(run_dirs)}).")
        return 2

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(run_dirs))
    rng.shuffle(idx)
    holdout_n = max(1, int(len(run_dirs) * args.holdout_frac))
    holdout_dirs = [run_dirs[i] for i in idx[:holdout_n]]

    # Load checkpoint and stats
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    with open(args.stats, "r", encoding="utf-8") as f:
        stats_raw: Dict[str, List[float]] = json.load(f)
    y_stats = {k: (v[0], v[1]) for k, v in stats_raw.items()}

    model = WeightsTransformer(
        token_size=int(ckpt["token_size"]),
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        nlayers=int(ckpt["nlayers"]),
        dim_feedforward=int(ckpt["d_ff"]),
        dropout=float(ckpt["dropout"]),
        out_dim=len(TARGET_FIELDS),
    )
    model.load_state_dict(ckpt["model_state"])  # type: ignore[arg-type]
    model.to(device).eval()

    # Prepare outputs
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir) / f"eval-{ts}"
    outdir.mkdir(parents=True, exist_ok=False)
    pred_csv = outdir / "predictions.csv"
    mae_json = outdir / "mae.json"

    fieldnames = ["run_dir"] + [f"true_{k}" for k in TARGET_FIELDS] + [f"pred_{k}" for k in TARGET_FIELDS]
    mae_sums = {k: 0.0 for k in TARGET_FIELDS}
    count = 0

    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rd in holdout_dirs:
            x, y = load_example(rd)
            # Standardize x per-example and tokenize like training
            xm, xs = x.mean(), x.std() + 1e-8
            xz = (x - xm) / xs
            token_size = int(ckpt["token_size"])
            L = int(np.ceil(len(xz) / token_size))
            pad_len = L * token_size - len(xz)
            if pad_len:
                xz = np.pad(xz, (0, pad_len), mode="constant")
            tokens = torch.from_numpy(xz.reshape(L, token_size).astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                y_pred = model(tokens.to(device)).squeeze(0).cpu().numpy()

            # Denormalize both y (which is raw here) and prediction (which is normalized)
            y_true = {k: float(y[k]) for k in TARGET_FIELDS}
            y_pred_denorm = denorm(y_pred, y_stats)

            row = {"run_dir": str(rd)}
            row.update({f"true_{k}": y_true[k] for k in TARGET_FIELDS})
            row.update({f"pred_{k}": y_pred_denorm[k] for k in TARGET_FIELDS})
            writer.writerow(row)

            for k in TARGET_FIELDS:
                mae_sums[k] += abs(y_true[k] - y_pred_denorm[k])
            count += 1

    maes = {k: (mae_sums[k] / max(count, 1)) for k in TARGET_FIELDS}
    with open(mae_json, "w", encoding="utf-8") as f:
        json.dump(maes, f, indent=2)

    print(f"Wrote predictions to {pred_csv}")
    print("Per-field MAE:")
    for k, v in maes.items():
        print(f"  {k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
