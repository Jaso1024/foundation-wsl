#!/usr/bin/env python3
"""
Unified CLI to launch:
- ModSum sweep data generation/training runs
- Foundation (weights->metadata) training on existing runs

By default, jobs launch in the background and stream logs to a timestamped file.
Use --fg to run in the foreground.

Examples:
  # Background sweep with defaults
  python scripts/cli.py sweep --runs 10000

  # Background foundation training on existing runs
  python scripts/cli.py foundation-train --epochs 50 --batch-size 1024

  # Foreground (no background)
  python scripts/cli.py --fg sweep --runs 1000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_bg(cmd: List[str], log_file: Path) -> int:
    ensure_dir(log_file.parent)
    with open(log_file, "ab", buffering=0) as lf:
        # Unbuffered (-u) in child to get timely logs
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    print(f"Started background job PID={proc.pid}\n  Log: {log_file}")
    return 0


def run_fg(cmd: List[str]) -> int:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fg", action="store_true", help="Run in foreground (default: background).")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ModSum + Foundation training CLI")
    ap.add_argument("--fg", action="store_true", help="Run in foreground (default: background).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Sweep subcommand
    sp = sub.add_parser("sweep", help="Run ModSum sweep (many runs)")
    sp.add_argument("--runs", type=int, default=200, help="Number of runs to execute. Default: 200.")
    sp.add_argument("--minN", type=int, default=7, help="Min N inclusive. Default: 7.")
    sp.add_argument("--maxN", type=int, default=31, help="Max N inclusive. Default: 31.")
    sp.add_argument("--base-out", type=str, default="runs/modsum", help="Per-run artifacts base dir.")
    sp.add_argument("--sweep-out", type=str, default="runs/modsum_sweeps", help="Sweep summary dir.")
    sp.add_argument("--seed", type=int, default=None, help="Sweep RNG seed.")
    # Optional perf knobs (only forwarded if underlying script supports them)
    sp.add_argument("--batch-size-fixed", type=int, default=None, help="Force batch size for all runs (if supported).")
    sp.add_argument("--batch-choices", type=str, default=None, help="CSV of batch sizes to sample (if supported).")
    sp.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per run (if supported).")
    sp.add_argument("--log", type=str, default=None, help="Log file path (default auto under sweep-out/cli_logs).")

    # Foundation training subcommand
    fp = sub.add_parser("foundation-train", help="Train weights->metadata foundation model on existing runs")
    fp.add_argument("--runs-base", type=str, default="runs/modsum", help="Directory with per-run artifacts.")
    fp.add_argument("--outdir", type=str, default="runs/weights2meta", help="Output directory for foundation model.")
    fp.add_argument("--epochs", type=int, default=30)
    fp.add_argument("--batch-size", type=int, default=512)
    fp.add_argument("--num-workers", type=int, default=0)
    fp.add_argument("--lr", type=float, default=1e-3)
    fp.add_argument("--token-size", type=int, default=64)
    fp.add_argument("--d-model", type=int, default=64)
    fp.add_argument("--nhead", type=int, default=4)
    fp.add_argument("--nlayers", type=int, default=2)
    fp.add_argument("--d-ff", type=int, default=128)
    fp.add_argument("--dropout", type=float, default=0.1)
    fp.add_argument("--train-frac", type=float, default=0.9)
    fp.add_argument("--seed", type=int, default=1337)
    fp.add_argument("--log", type=str, default=None, help="Log file path (default auto under outdir/cli_logs).")

    return ap


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = build_parser()
    args = ap.parse_args(argv)

    # Decide foreground/background
    foreground = bool(args.fg)

    if args.cmd == "sweep":
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "experiments" / "sweep_modsum.py"),
            f"--runs={args.runs}",
            f"--minN={args.minN}",
            f"--maxN={args.maxN}",
            f"--base-out={args.base_out}",
            f"--sweep-out={args.sweep_out}",
        ]
        if args.seed is not None:
            cmd.append(f"--seed={args.seed}")
        # Forward optional flags only if provided
        if args.batch_size_fixed is not None:
            cmd.append(f"--batch-size-fixed={args.batch_size_fixed}")
        if args.batch_choices:
            cmd.append(f"--batch-choices={args.batch_choices}")
        if args.num_workers is not None:
            cmd.append(f"--num-workers={args.num_workers}")

        log = Path(args.log) if args.log else (REPO_ROOT / args.sweep_out / "cli_logs" / f"sweep_{ts()}.out")
        return run_fg(cmd) if foreground else run_bg(cmd, log)

    if args.cmd == "foundation-train":
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "experiments" / "weights2meta_train.py"),
            f"--runs-base={args.runs_base}",
            f"--outdir={args.outdir}",
            f"--epochs={args.epochs}",
            f"--batch-size={args.batch_size}",
            f"--num-workers={args.num_workers}",
            f"--lr={args.lr}",
            f"--token-size={args.token_size}",
            f"--d-model={args.d_model}",
            f"--nhead={args.nhead}",
            f"--nlayers={args.nlayers}",
            f"--d-ff={args.d_ff}",
            f"--dropout={args.dropout}",
            f"--train-frac={args.train_frac}",
            f"--seed={args.seed}",
        ]
        log = Path(args.log) if args.log else (REPO_ROOT / args.outdir / "cli_logs" / f"foundation_train_{ts()}.out")
        return run_fg(cmd) if foreground else run_bg(cmd, log)

    ap.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

