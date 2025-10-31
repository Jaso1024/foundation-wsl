#!/usr/bin/env python3
"""
Train a small Transformer to map saved model weights -> run metadata.

Input: state_dict from runs under a base directory (e.g., runs/modsum/*/model.pt)
Targets: selected config + metrics from each run's config.json

We flatten model weights, split into fixed-size tokens, project to d_model,
apply a tiny TransformerEncoder, pool, then predict metadata via an MLP head.

This is CPU-friendly and meant for quick experimentation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


TARGET_FIELDS = [
    # Config-derived
    "N",
    "epochs",
    "lr",
    "emb_dim",
    "hidden_dim",
    "batch_size",
    "train_frac",
    # Metrics-derived
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
]


def list_run_dirs(base: Path) -> List[Path]:
    base = base.expanduser().resolve()
    if not base.exists():
        return []
    return sorted([p for p in base.glob("*_N*_seed*") if (p / "model.pt").exists() and (p / "config.json").exists()])


def load_example(run_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    flat_parts: List[np.ndarray] = []
    for k, t in state.items():
        if not isinstance(t, torch.Tensor):
            continue
        flat_parts.append(t.detach().cpu().contiguous().view(-1).numpy())
    # NumPy < 2.0 doesn't support dtype= on concatenate; cast afterward
    flat = np.concatenate(flat_parts).astype(np.float32)

    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    cfg = info["config"]
    met = info["metrics"]

    y: Dict[str, float] = {
        "N": float(cfg["N"]),
        "epochs": float(cfg["epochs"]),
        "lr": float(cfg["lr"]),
        "emb_dim": float(cfg["emb_dim"]),
        "hidden_dim": float(cfg.get("hidden_dim", 32.0)),
        "batch_size": float(cfg["batch_size"]),
        "train_frac": float(cfg["train_frac"]),
        "train_loss": float(met["train_loss"]),
        "train_accuracy": float(met["train_accuracy"]),
        "test_loss": float(met["test_loss"]),
        "test_accuracy": float(met["test_accuracy"]),
    }
    return flat, y


class WeightsMetaDataset(Dataset):
    def __init__(self, run_dirs: List[Path], token_size: int, y_stats: Dict[str, Tuple[float, float]] | None = None):
        self.X: List[np.ndarray] = []
        self.Y: List[np.ndarray] = []
        for rd in run_dirs:
            x, y = load_example(rd)
            self.X.append(x)
            self.Y.append(np.array([y[k] for k in TARGET_FIELDS], dtype=np.float32))

        self.X = [x.astype(np.float32) for x in self.X]

        if y_stats is None:
            y_arr = np.stack(self.Y)
            self.y_mean = y_arr.mean(axis=0)
            self.y_std = y_arr.std(axis=0) + 1e-8
        else:
            self.y_mean = np.array([y_stats[k][0] for k in TARGET_FIELDS], dtype=np.float32)
            self.y_std = np.array([y_stats[k][1] for k in TARGET_FIELDS], dtype=np.float32)

        self.token_size = token_size

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        # Standardize weights per-example to remove scale; then chunk.
        xm = x.mean()
        xs = x.std() + 1e-8
        xz = (x - xm) / xs
        # Chunk into tokens of size token_size with zero-pad at end
        L = int(np.ceil(len(xz) / self.token_size))
        pad_len = L * self.token_size - len(xz)
        if pad_len:
            xz = np.pad(xz, (0, pad_len), mode="constant")
        tokens = xz.reshape(L, self.token_size).astype(np.float32)

        y = (self.Y[idx] - self.y_mean) / self.y_std
        return torch.from_numpy(tokens), torch.from_numpy(y)

    def y_norm_stats(self) -> Dict[str, Tuple[float, float]]:
        return {k: (float(m), float(s)) for k, m, s in zip(TARGET_FIELDS, self.y_mean, self.y_std)}


class WeightsTransformer(nn.Module):
    def __init__(self, token_size: int, d_model: int, nhead: int, nlayers: int, dim_feedforward: int, dropout: float, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(token_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.proj(tokens)
        h = self.encoder(h)
        # Global mean pool over token dimension
        h = h.mean(dim=1)
        out = self.head(h)
        return out


@dataclass
class TrainArgs:
    runs_base: str
    outdir: str
    epochs: int
    batch_size: int
    lr: float
    token_size: int
    d_model: int
    nhead: int
    nlayers: int
    d_ff: int
    dropout: float
    train_frac: float
    seed: int


def parse_args(argv: List[str] | None = None) -> TrainArgs:
    ap = argparse.ArgumentParser(description="Train transformer to predict run metadata from weights.")
    ap.add_argument("--runs-base", type=str, default="runs/modsum", help="Directory with individual runs (containing model.pt, config.json).")
    ap.add_argument("--outdir", type=str, default="runs/weights2meta", help="Where to save the transformer and summary.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--token-size", type=int, default=64, help="Number of weight scalars per token before projection.")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args(argv)
    return TrainArgs(**vars(a))


def set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch, device):
    tokens, y = batch
    return tokens.to(device), y.to(device)


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    # batch of variable-length token sequences [Li, token_size]
    ys = torch.stack([y for _, y in batch], dim=0)
    lengths = [t.size(0) for t, _ in batch]
    max_len = max(lengths)
    token_size = batch[0][0].size(1)
    out = torch.zeros(len(batch), max_len, token_size, dtype=batch[0][0].dtype)
    for i, (t, _) in enumerate(batch):
        out[i, : t.size(0)] = t
    return out, ys


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    set_seeds(args.seed)

    base = Path(args.runs_base)
    run_dirs = list_run_dirs(base)
    if len(run_dirs) < 10:
        print(f"Not enough runs found under {base} (found {len(run_dirs)}). Generate more with sweep_modsum.py.")
        return 2

    # Split train/val
    n_train = int(len(run_dirs) * args.train_frac)
    train_dirs = run_dirs[:n_train]
    val_dirs = run_dirs[n_train:]

    train_ds = WeightsMetaDataset(train_dirs, token_size=args.token_size)
    y_stats = train_ds.y_norm_stats()
    val_ds = WeightsMetaDataset(val_dirs, token_size=args.token_size, y_stats=y_stats)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate, pin_memory=pin)

    model = WeightsTransformer(
        token_size=args.token_size,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dim_feedforward=args.d_ff,
        dropout=args.dropout,
        out_dim=len(TARGET_FIELDS),
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    def run_epoch(loader, train: bool) -> float:
        total = 0.0
        n = 0
        if train:
            model.train()
        else:
            model.eval()
        for batch in loader:
            tokens, y = to_device(batch, device)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                pred = model(tokens)
                loss = crit(pred, y)
                if train:
                    loss.backward()
                    opt.step()
            bs = tokens.size(0)
            total += loss.item() * bs
            n += bs
        return total / max(n, 1)

    best_val = float("inf")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "weights2meta.pt"
    stats_path = outdir / "y_norm.json"
    meta_path = outdir / "meta.json"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        print(f"epoch {epoch:03d} | train_mse {tr:.4f} | val_mse {va:.4f}")
        if va < best_val:
            best_val = va
            torch.save({
                "model_state": model.state_dict(),
                "token_size": args.token_size,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "nlayers": args.nlayers,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "targets": TARGET_FIELDS,
            }, ckpt_path)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump({k: list(v) for k, v in y_stats.items()}, f, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)
    print(f"Saved best checkpoint to {ckpt_path} with val_mse={best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
