#!/usr/bin/env python3
"""
Interactive CLI to:
- Run ModSum sweep (many per-run trainings)
- Train the foundation (weights->metadata) model on existing runs
- List and kill background jobs launched via this CLI

Defaults:
- Runs jobs in the background and writes logs to timestamped files
- Returns the PID on launch so you can kill it later
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = REPO_ROOT / "runs" / "cli"
REG_PATH = REG_DIR / "registry.json"


def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_registry() -> Dict[str, Dict]:
    try:
        with open(REG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(reg: Dict[str, Dict]) -> None:
    ensure_dir(REG_DIR)
    tmp = REG_DIR / f"registry.{ts()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    tmp.replace(REG_PATH)


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_bg(cmd: List[str], log_file: Path, job_type: str, params: Dict[str, str]) -> int:
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
    # Register job
    reg = load_registry()
    jid = str(uuid.uuid4())
    reg[jid] = {
        "id": jid,
        "pid": proc.pid,
        "pgid": proc.pid,
        "cmd": cmd,
        "type": job_type,
        "params": params,
        "log": str(log_file),
        "cwd": str(REPO_ROOT),
        "start": ts(),
        "status": "running",
    }
    save_registry(reg)
    print(f"Started background job\n  PID: {proc.pid}\n  Log: {log_file}")
    return 0


def run_fg(cmd: List[str]) -> int:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fg", action="store_true", help="Run in foreground (default: background).")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ModSum + Foundation training CLI", add_help=True)
    ap.add_argument("--fg", action="store_true", help="Run in foreground (default: background).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Sweep subcommand
    sp = sub.add_parser("sweep", help="Run ModSum sweep (many runs)")
    sp.add_argument("--runs", type=int, default=200, help="Number of runs to execute. Default: 200.")
    sp.add_argument("--minN", type=int, default=7, help="Min N inclusive. Default: 7.")
    sp.add_argument("--maxN", type=int, default=31, help="Max N inclusive. Default: 31.")
    sp.add_argument("--epochs", type=int, default=20, help="Epochs per run. Default: 20.")
    sp.add_argument("--batch-size", type=int, default=128, help="Batch size per run. Default: 128.")
    sp.add_argument("--base-out", type=str, default="runs/modsum", help="Per-run artifacts base dir.")
    sp.add_argument("--sweep-out", type=str, default="runs/modsum_sweeps", help="Sweep summary dir.")
    sp.add_argument("--seed", type=int, default=None, help="Sweep RNG seed.")
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


def prompt_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        print("Invalid int; using default.")
        return default


def prompt_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        print("Invalid float; using default.")
        return default


def prompt_str(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s or default


def prompt_yes(prompt: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    s = input(f"{prompt} ({default}): ").strip().lower()
    if not s:
        return default_yes
    return s in {"y", "yes"}


def interactive_menu() -> int:
    while True:
        print("\n=== ModSum/Foundation CLI ===")
        print("1) Run ModSum sweep")
        print("2) Train Foundation (weights->metadata)")
        print("3) List background jobs")
        print("4) Kill background jobs")
        print("5) Exit")
        choice = input("Select [1-5]: ").strip()
        if choice == "1":
            return interactive_sweep()
        elif choice == "2":
            return interactive_foundation_train()
        elif choice == "3":
            return list_jobs()
        elif choice == "4":
            return kill_jobs()
        elif choice == "5":
            return 0
        else:
            print("Invalid choice.")


def build_sweep_cmd(params: Dict[str, str]) -> List[str]:
    script = REPO_ROOT / "experiments" / "sweep_modsum.py"

    def supports(flag: str) -> bool:
        try:
            out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
            return flag in (out.stdout or "") or flag in (out.stderr or "")
        except Exception:
            return False

    cmd = [
        sys.executable,
        "-u",
        str(script),
        f"--runs={params['runs']}",
        f"--minN={params['minN']}",
        f"--maxN={params['maxN']}",
        f"--epochs={params['epochs']}",
        f"--batch-size={params['batch_size']}",
        f"--base-out={params['base_out']}",
        f"--sweep-out={params['sweep_out']}",
    ]
    if params.get("seed"):
        cmd.append(f"--seed={params['seed']}")
    return cmd


def interactive_sweep() -> int:
    print("\n[ModSum Sweep]")
    runs = prompt_int("Number of runs", 200)
    minN = prompt_int("Min N", 7)
    maxN = prompt_int("Max N", 31)
    epochs = prompt_int("Epochs per run", 20)
    batch_size = prompt_int("Batch size per run", 128)
    base_out = prompt_str("Base out dir", "runs/modsum")
    sweep_out = prompt_str("Sweep out dir", "runs/modsum_sweeps")
    seed_s = prompt_str("Seed (blank=random)", "")
    bg = prompt_yes("Run in background?", True)

    params = {
        "runs": str(runs),
        "minN": str(minN),
        "maxN": str(maxN),
        "base_out": base_out,
        "sweep_out": sweep_out,
        "seed": seed_s,
        "epochs": str(epochs),
        "batch_size": str(batch_size),
    }
    cmd = build_sweep_cmd(params)
    log = REPO_ROOT / sweep_out / "cli_logs" / f"sweep_{ts()}.out"
    if bg:
        return run_bg(cmd, log, job_type="sweep", params=params)
    else:
        return run_fg(cmd)


def build_foundation_cmd(params: Dict[str, str]) -> List[str]:
    script = REPO_ROOT / "experiments" / "weights2meta_train.py"

    def supports(flag: str) -> bool:
        try:
            out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
            return flag in (out.stdout or "") or flag in (out.stderr or "")
        except Exception:
            return False

    cmd = [
        sys.executable,
        "-u",
        str(script),
        f"--runs-base={params['runs_base']}",
        f"--outdir={params['outdir']}",
        f"--epochs={params['epochs']}",
        f"--batch-size={params['batch_size']}",
        # optional flags appended below
        f"--lr={params['lr']}",
        f"--token-size={params['token_size']}",
        f"--d-model={params['d_model']}",
        f"--nhead={params['nhead']}",
        f"--nlayers={params['nlayers']}",
        f"--d-ff={params['d_ff']}",
        f"--dropout={params['dropout']}",
        f"--train-frac={params['train_frac']}",
        f"--seed={params['seed']}",
    ]
    if params.get("num_workers") is not None and supports("--num-workers"):
        cmd.append(f"--num-workers={params['num_workers']}")
    return cmd


def interactive_foundation_train() -> int:
    print("\n[Foundation Training]")
    runs_base = prompt_str("Runs base dir", "runs/modsum")
    outdir = prompt_str("Output dir", "runs/weights2meta")
    epochs = prompt_int("Epochs", 30)
    batch_size = prompt_int("Batch size", 512)
    num_workers = prompt_int("DataLoader workers", 0)
    lr = prompt_float("Learning rate", 1e-3)
    token_size = prompt_int("Token size", 64)
    d_model = prompt_int("d_model", 64)
    nhead = prompt_int("nhead", 4)
    nlayers = prompt_int("nlayers", 2)
    d_ff = prompt_int("d_ff", 128)
    dropout = prompt_float("dropout", 0.1)
    train_frac = prompt_float("Train fraction", 0.9)
    seed = prompt_int("Seed", 1337)
    bg = prompt_yes("Run in background?", True)

    params = {
        "runs_base": runs_base,
        "outdir": outdir,
        "epochs": str(epochs),
        "batch_size": str(batch_size),
        "num_workers": str(num_workers),
        "lr": str(lr),
        "token_size": str(token_size),
        "d_model": str(d_model),
        "nhead": str(nhead),
        "nlayers": str(nlayers),
        "d_ff": str(d_ff),
        "dropout": str(dropout),
        "train_frac": str(train_frac),
        "seed": str(seed),
    }
    cmd = build_foundation_cmd(params)
    log = REPO_ROOT / outdir / "cli_logs" / f"foundation_train_{ts()}.out"
    if bg:
        return run_bg(cmd, log, job_type="foundation-train", params=params)
    else:
        return run_fg(cmd)


def list_jobs() -> int:
    reg = load_registry()
    if not reg:
        print("No jobs tracked.")
        return 0
    print("\nTracked jobs:")
    rows = []
    for i, j in enumerate(reg.values(), 1):
        alive = is_running(int(j.get("pid", -1)))
        status = "running" if alive else "stopped"
        print(f"{i:3d}. PID={j['pid']:>6} type={j['type']:<16} started={j['start']} status={status} log={j['log']}")
        rows.append(j)
    return 0


def kill_jobs() -> int:
    reg = load_registry()
    if not reg:
        print("No jobs tracked.")
        return 0
    jobs = list(reg.values())
    print("\nSelect jobs to kill (comma-separated indices), or 'all':")
    for i, j in enumerate(jobs, 1):
        alive = is_running(int(j.get("pid", -1)))
        status = "running" if alive else "stopped"
        print(f"{i:3d}. PID={j['pid']:>6} type={j['type']:<16} started={j['start']} status={status} log={j['log']}")
    s = input("Kill which? [e.g., 1,3] or all: ").strip().lower()
    idxs: List[int] = []
    if s == "all":
        idxs = list(range(1, len(jobs) + 1))
    else:
        try:
            idxs = [int(x) for x in s.split(",") if x.strip()]
        except ValueError:
            print("Invalid selection.")
            return 1
    if not idxs:
        print("No selection.")
        return 1
    grace = prompt_int("Grace seconds before SIGKILL", 5)
    for i in idxs:
        if not (1 <= i <= len(jobs)):
            continue
        j = jobs[i - 1]
        pid = int(j.get("pid", -1))
        if pid <= 0:
            continue
        try:
            print(f"Sending SIGTERM to PID {pid} ...")
            os.killpg(pid, signal.SIGTERM)
        except Exception as e:
            print(f"  WARN: SIGTERM failed for {pid}: {e}")
        t0 = time.time()
        while time.time() - t0 < grace:
            if not is_running(pid):
                break
            time.sleep(0.2)
        if is_running(pid):
            try:
                print(f"  Escalating SIGKILL to PID {pid} ...")
                os.killpg(pid, signal.SIGKILL)
            except Exception as e:
                print(f"  WARN: SIGKILL failed for {pid}: {e}")
        # Update registry status
        j["status"] = "stopped"
    # Save registry
    new_reg = {j["id"]: j for j in jobs}
    save_registry(new_reg)
    print("Done.")
    return 0


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    # If subcommands are provided, keep non-interactive mode for power users
    if argv:
        ap = build_parser()
        args = ap.parse_args(argv)
        # Decide foreground/background
        foreground = bool(args.fg)

        if args.cmd == "sweep":
            params = {
                "runs": str(args.runs),
                "minN": str(args.minN),
                "maxN": str(args.maxN),
                "epochs": str(args.epochs),
                "batch_size": str(args.batch_size),
                "base_out": args.base_out,
                "sweep_out": args.sweep_out,
                "seed": str(args.seed) if args.seed is not None else "",
            }
            cmd = build_sweep_cmd(params)
            log = Path(args.log) if args.log else (REPO_ROOT / args.sweep_out / "cli_logs" / f"sweep_{ts()}.out")
            return run_fg(cmd) if foreground else run_bg(cmd, log, job_type="sweep", params=params)

        if args.cmd == "foundation-train":
            params = {
                "runs_base": args.runs_base,
                "outdir": args.outdir,
                "epochs": str(args.epochs),
                "batch_size": str(args.batch_size),
                "num_workers": str(args.num_workers),
                "lr": str(args.lr),
                "token_size": str(args.token_size),
                "d_model": str(args.d_model),
                "nhead": str(args.nhead),
                "nlayers": str(args.nlayers),
                "d_ff": str(args.d_ff),
                "dropout": str(args.dropout),
                "train_frac": str(args.train_frac),
                "seed": str(args.seed),
            }
            cmd = build_foundation_cmd(params)
            log = Path(args.log) if args.log else (REPO_ROOT / args.outdir / "cli_logs" / f"foundation_train_{ts()}.out")
            return run_fg(cmd) if foreground else run_bg(cmd, log, job_type="foundation-train", params=params)

        ap.error("Unknown command")
        return 2

    # No args -> interactive menu
    return interactive_menu()


if __name__ == "__main__":
    raise SystemExit(main())
