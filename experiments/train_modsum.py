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
import torch.nn.functional as F
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
    train_samples: int | None
    resample_each_epoch: bool
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


class ModSumMLP(nn.Module):
    """Regular MLP over one-hot(x) || one-hot(y) -> N-class logits."""

    def __init__(self, N: int, hidden1: int = 32, hidden2: int = 128):
        super().__init__()
        self.N = N
        inp = 2 * N
        self.net = nn.Sequential(
            nn.Linear(inp, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, N),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x1 = F.one_hot(x, num_classes=self.N).to(dtype=torch.float32)
        y1 = F.one_hot(y, num_classes=self.N).to(dtype=torch.float32)
        z = torch.cat([x1, y1], dim=1)
        return self.net(z)


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
    p.add_argument("--lr", type=float, default=3e-3, help="Learning rate (Adam). Default: 3e-3.")
    p.add_argument("--emb-dim", type=int, default=32, help="First hidden width. Default: 32.")
    p.add_argument("--hidden-dim", type=int, default=128, help="Second hidden width. Default: 128.")
    p.add_argument("--train-frac", type=float, default=0.8, help="Fraction of total pairs reserved as train pool (test is the rest). Default: 0.8.")
    p.add_argument("--train-samples", type=int, default=None, help="Number of training samples per epoch (<= train pool). Default: use full train pool.")
    p.add_argument("--resample-each-epoch", action="store_true", help="If set with --train-samples, resample a new subset each epoch.")
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

    # Build full pool of samples; derive a fixed test split, and a train pool
    all_samples = list(gen_all_pairs(N))
    rng = random.Random(seed)
    idx = list(range(len(all_samples)))
    rng.shuffle(idx)
    cut = int(args.train_frac * len(all_samples))
    train_pool_idx, test_idx = idx[:cut], idx[cut:]
    test_samples = [all_samples[i] for i in test_idx]

    # Helper to make a train dataset for a given epoch
    def make_train_ds(epoch: int) -> ModSumDataset:
        if args.train_samples is None or args.train_samples >= len(train_pool_idx):
            chosen = train_pool_idx
        else:
            # Stable but different sample each epoch when resampling: shift RNG by epoch
            rng_e = random.Random(seed + epoch)
            chosen = rng_e.sample(train_pool_idx, k=args.train_samples)
        return ModSumDataset([all_samples[i] for i in chosen])

    # Initial datasets / loaders
    train_ds = make_train_ds(epoch=0)
    test_ds = ModSumDataset(test_samples)
    device = get_device()
    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=pin)

    # Model
    model = ModSumMLP(N=N, hidden1=args.emb_dim, hidden2=args.hidden_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train (optionally resample training subset each epoch)
    for ep in range(args.epochs):
        if getattr(args, "resample_each_epoch", False) and args.train_samples is not None:
            train_ds = make_train_ds(epoch=ep + 1)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
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

    # Save datasets used (store the last epoch's train subset for reference)
    # Reconstruct Sample objects from the final train dataset's tensors
    train_samples_last = [
        Sample(int(x), int(y), int(t))
        for x, y, t in zip(train_ds.x.tolist(), train_ds.y.tolist(), train_ds.t.tolist())
    ]
    with open(run_dir / "train.csv", "w", newline="", encoding="utf-8") as f:
        write_csv(train_samples_last, f)
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
        train_samples=int(args.train_samples) if args.train_samples is not None else None,
        resample_each_epoch=bool(getattr(args, "resample_each_epoch", False)),
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
