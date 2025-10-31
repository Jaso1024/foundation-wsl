#!/usr/bin/env python3
"""
Train a tiny CPU model to learn x + y (mod N) on a synthetic dataset.

Artifacts are saved under: runs/modsum/<timestamp>_N{N}_seed{seed}/

Saved files:
- config.json: task/config + metrics (train/test loss & accuracy)
- model.pt: PyTorch state_dict
- train.csv, test.csv: datasets used for this run

Defaults are chosen to finish in < 30s on CPU for typical Ns.

Example:
  python experiments/train_modsum.py               # random N in [7, 31]
  python experiments/train_modsum.py --N 13        # fixed N
  python experiments/train_modsum.py --minN 11 --maxN 25 --epochs 30 --lr 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

# Local imports (dataset generator)
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from scripts.modsum_dataset import Sample, gen_all_pairs, write_csv  # type: ignore


def try_import_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        import torch.optim as optim  # noqa: F401
        return True
    except Exception as e:
        print("PyTorch import failed. Please install torch: pip install torch", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return False


if not try_import_torch():
    sys.exit(2)

import torch
import torch.nn as nn
import torch.optim as optim


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    N: int
    seed: int
    epochs: int
    lr: float
    emb_dim: int
    hidden_dim: int
    train_frac: float
    batch_size: int
    all_pairs: bool
    run_dir: str


class ModSumDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.x = torch.tensor([s.x for s in samples], dtype=torch.long)
        self.y = torch.tensor([s.y for s in samples], dtype=torch.long)
        self.t = torch.tensor([s.target for s in samples], dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.t[idx]


class TinyModNet(nn.Module):
    """Two embeddings -> combine -> nonlinearity -> classifier over N classes.

    Intentionally tiny and CPU-friendly, but with a hidden nonlinearity.
    """

    def __init__(self, N: int, emb_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.N = N
        self.emb_x = nn.Embedding(N, emb_dim)
        self.emb_y = nn.Embedding(N, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, N)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ex = self.emb_x(x)
        ey = self.emb_y(y)
        h = ex + ey
        h = self.fc1(h)
        h = self.act(h)
        logits = self.fc2(h)
        return logits


def split_train_test(samples: List[Sample], train_frac: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    cut = int(train_frac * len(samples))
    train_idx, test_idx = idx[:cut], idx[cut:]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test


def train_one_epoch(model: nn.Module, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y, t in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, y)
        loss = criterion(logits, t)
        loss.backward()
        optimizer.step()
        bs = x.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss_acc(model: nn.Module, loader, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y, t in loader:
        logits = model(x, y)
        loss = criterion(logits, t)
        pred = logits.argmax(dim=1)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        correct += (pred == t).sum().item()
        n += bs
    return total_loss / max(n, 1), correct / max(n, 1)


def build_run_dir(base_dir: Path, N: int, seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"{ts}_N{N}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tiny model on x + y mod N.")
    p.add_argument("--N", type=int, default=None, help="Use a fixed N. If omitted, sample uniformly in [--minN, --maxN].")
    p.add_argument("--minN", type=int, default=7, help="Min N when sampling (inclusive). Default: 7.")
    p.add_argument("--maxN", type=int, default=31, help="Max N when sampling (inclusive). Default: 31.")
    p.add_argument("--seed", type=int, default=None, help="Base seed. If omitted, chosen randomly.")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs. Default: 20.")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate. Default: 0.1.")
    p.add_argument("--emb-dim", type=int, default=16, help="Embedding dimension. Default: 16.")
    p.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension. Default: 32.")
    p.add_argument("--train-frac", type=float, default=0.8, help="Train split fraction. Default: 0.8.")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size. Default: 128.")
    p.add_argument("--all-pairs", action="store_true", help="Use all N^2 pairs (recommended). If not set, will still use all-pairs for speed & coverage.")
    p.add_argument("--outdir", type=str, default="runs/modsum", help="Base output directory. Default: runs/modsum")
    return p.parse_args(argv)


def run_training(args) -> Path:
    """Run a single training using parsed args and return the created run_dir.

    This splits out the core work so sweep drivers can get the exact run_dir
    without relying on filesystem mtimes.
    """

    N = args.N if args.N is not None else random.randint(args.minN, args.maxN)
    seed = args.seed if args.seed is not None else random.randint(1, 10_000_000)
    set_seeds(seed)

    # Generate dataset (all pairs by default for good coverage and speed)
    samples = list(gen_all_pairs(N))

    # Split
    train_samples, test_samples = split_train_test(samples, args.train_frac, seed)

    # Datasets / loaders
    train_ds = ModSumDataset(train_samples)
    test_ds = ModSumDataset(test_samples)
    device = get_device()
    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=pin)

    # Model
    model = TinyModNet(N=N, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Train
    for _ in range(args.epochs):
        # move batches to device inside train loop
        model.train()
        total_loss = 0.0
        n = 0
        for x, y, t in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, y)
            loss = criterion(logits, t)
            loss.backward()
            optimizer.step()
            bs = x.shape[0]
            total_loss += loss.item() * bs
            n += bs

    # Final metrics
    # evaluation with device tensors
    @torch.no_grad()
    def eval_loader(loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        for x, y, t in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            logits = model(x, y)
            loss = criterion(logits, t)
            pred = logits.argmax(dim=1)
            bs = x.shape[0]
            total_loss += loss.item() * bs
            correct += (pred == t).sum().item()
            n += bs
        return total_loss / max(n, 1), correct / max(n, 1)

    train_loss, train_acc = eval_loader(train_loader)
    test_loss, test_acc = eval_loader(test_loader)

    # Save artifacts
    base_dir = Path(args.outdir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_dir(base_dir, N=N, seed=seed)

    # Save datasets used
    with open(run_dir / "train.csv", "w", newline="", encoding="utf-8") as f:
        write_csv(train_samples, f)
    with open(run_dir / "test.csv", "w", newline="", encoding="utf-8") as f:
        write_csv(test_samples, f)

    # Save model
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Save config + metrics
    cfg = RunConfig(
        N=N,
        seed=seed,
        epochs=args.epochs,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        all_pairs=True,
        run_dir=str(run_dir),
    )

    info = {
        "task": "x_plus_y_mod_N",
        "config": asdict(cfg),
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
    return run_dir


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    _ = run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
