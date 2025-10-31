#!/usr/bin/env python3
"""
Train a simple 2-layer MLP on MNIST.

Artifacts are saved under: runs/mnist/<timestamp>_seed{seed}/
 - config.json: config + metrics (train/test loss & accuracy)
 - model.pt: PyTorch state_dict

CPU-friendly; uses CUDA if available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seeds(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    seed: int
    epochs: int
    lr: float
    batch_size: int
    hidden1: int
    hidden2: int
    val_frac: float
    run_dir: str


class MNISTMLP(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.view(x.size(0), -1)
        return self.net(z)


def build_run_dir(base: Path, seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rd = base / f"{ts}_seed{seed}"
    rd.mkdir(parents=True, exist_ok=False)
    return rd


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP on MNIST.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train used for validation.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--outdir", type=str, default="runs/mnist")
    p.add_argument("--data", type=str, default="data", help="Dataset cache directory.")
    return p.parse_args(argv)


@torch.no_grad()
def eval_loss_acc(model: nn.Module, loader, criterion) -> Tuple[float, float]:
    model.eval()
    total = 0.0
    correct = 0
    n = 0
    for x, t in loader:
        logits = model(x)
        loss = criterion(logits, t)
        pred = logits.argmax(dim=1)
        bs = x.size(0)
        total += loss.item() * bs
        correct += (pred == t).sum().item()
        n += bs
    return total / max(n, 1), correct / max(n, 1)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    set_seeds(args.seed)
    dev = device()

    # Data
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    def manual_download(dest_root: Path) -> None:
        import ssl
        from urllib.request import urlopen
        base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]
        raw = dest_root / "MNIST" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        ctx = ssl.create_default_context()
        # Best-effort: if corporate MITM breaks chain, fall back to unverified context
        try_unverified = False
        try:
            for fn in files:
                with urlopen(base + fn, context=ctx) as r, open(raw / fn, "wb") as f:
                    f.write(r.read())
        except Exception:
            try_unverified = True
        if try_unverified:
            unverified = ssl._create_unverified_context()  # type: ignore[attr-defined]
            for fn in files:
                with urlopen(base + fn, context=unverified) as r, open(raw / fn, "wb") as f:
                    f.write(r.read())

    try:
        train_full = datasets.MNIST(root=args.data, train=True, transform=tfm, download=True)
    except Exception:
        # Try a manual fetch from the OSSCI mirror (with SSL fallback), then load without download
        manual_download(Path(args.data))
        train_full = datasets.MNIST(root=args.data, train=True, transform=tfm, download=False)
    val_n = int(len(train_full) * args.val_frac)
    train_n = len(train_full) - val_n
    train_ds, val_ds = random_split(train_full, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))
    try:
        test_ds = datasets.MNIST(root=args.data, train=False, transform=tfm, download=True)
    except Exception:
        test_ds = datasets.MNIST(root=args.data, train=False, transform=tfm, download=False)

    pin = dev.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=pin)

    # Model
    model = MNISTMLP(hidden1=args.hidden1, hidden2=args.hidden2).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(loader, train: bool) -> float:
        total = 0.0
        n = 0
        model.train() if train else model.eval()
        for x, t in loader:
            x, t = x.to(dev, non_blocking=pin), t.to(dev, non_blocking=pin)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = crit(logits, t)
                if train:
                    loss.backward()
                    opt.step()
            bs = x.size(0)
            total += loss.item() * bs
            n += bs
        return total / max(n, 1)

    best_val = float("inf")
    for _ in range(args.epochs):
        run_epoch(train_loader, train=True)
        val_mse = run_epoch(val_loader, train=False)
        if val_mse < best_val:
            best_val = val_mse

    # Metrics
    train_loss, train_acc = eval_loss_acc(model, train_loader, crit)
    test_loss, test_acc = eval_loss_acc(model, test_loader, crit)

    # Save
    base = Path(args.outdir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_dir(base, args.seed)
    torch.save(model.state_dict(), run_dir / "model.pt")
    info = {
        "task": "mnist_mlp",
        "config": asdict(RunConfig(
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            val_frac=args.val_frac,
            run_dir=str(run_dir),
        )),
        "metrics": {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        },
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Saved run to: {run_dir}")
    print(json.dumps(info["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
