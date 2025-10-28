#!/usr/bin/env python3
"""
Generate datasets for x + y (mod N).

Features:
- Uniform random pairs or complete all-pairs grid
- CSV or JSONL output
- Optional fixed random seed for reproducibility

Examples:
  python scripts/modsum_dataset.py --N 7 -n 100 --seed 42 -o data.csv
  python scripts/modsum_dataset.py --N 13 --all-pairs -f jsonl -o data.jsonl
  python scripts/modsum_dataset.py --N 10 -n 20  # writes CSV to stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator, List, Tuple


@dataclass(frozen=True)
class Sample:
    x: int
    y: int
    target: int  # (x + y) % N


def gen_all_pairs(N: int) -> Iterator[Sample]:
    for x, y in product(range(N), repeat=2):
        yield Sample(x=x, y=y, target=(x + y) % N)


def gen_random_pairs(N: int, size: int, unique: bool, rng: random.Random) -> Iterator[Sample]:
    if unique:
        total = N * N
        if size > total:
            raise ValueError(f"Requested size {size} exceeds number of unique pairs {total} when unique sampling is enabled.")
        # Sample without replacement over the N^2 grid by indexing pairs
        # Map index k in [0, N^2) to (x, y) = (k // N, k % N)
        indices = rng.sample(range(N * N), k=size)
        for k in indices:
            x, y = divmod(k, N)
            yield Sample(x=x, y=y, target=(x + y) % N)
    else:
        for _ in range(size):
            x = rng.randrange(N)
            y = rng.randrange(N)
            yield Sample(x=x, y=y, target=(x + y) % N)


def write_csv(samples: Iterable[Sample], out_fp) -> None:
    writer = csv.writer(out_fp)
    writer.writerow(["x", "y", "target"])  # header
    for s in samples:
        writer.writerow([s.x, s.y, s.target])


def write_jsonl(samples: Iterable[Sample], out_fp) -> None:
    for s in samples:
        out_fp.write(json.dumps({"x": s.x, "y": s.y, "target": s.target}) + "\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate x + y (mod N) datasets.")
    p.add_argument("--N", type=int, required=True, help="Modulus N (positive integer).")

    size_group = p.add_mutually_exclusive_group()
    size_group.add_argument("-n", "--size", type=int, help="Number of random samples to generate.")
    size_group.add_argument("--all-pairs", action="store_true", help="Generate all N^2 pairs (0..N-1)^2.")

    p.add_argument("--unique", action="store_true", help="When sampling, ensure unique (x,y) pairs; at most N^2.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (only used for random sampling).")
    p.add_argument("-o", "--out", default="-", help="Output file path or '-' for stdout. Default: '-'.")
    p.add_argument("-f", "--format", choices=["csv", "jsonl"], default="csv", help="Output format. Default: csv.")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.N is None or args.N <= 0:
        print("Error: --N must be a positive integer.", file=sys.stderr)
        return 2

    if args.all_pairs:
        if args.size is not None:
            print("Error: specify either --all-pairs or --size, not both.", file=sys.stderr)
            return 2
        samples_iter = gen_all_pairs(args.N)
    else:
        if args.size is None:
            print("Error: either --size must be provided or use --all-pairs.", file=sys.stderr)
            return 2
        rng = random.Random(args.seed)
        samples_iter = gen_random_pairs(args.N, args.size, args.unique, rng)

    # Prepare output
    out_fp = None
    close_after = False
    try:
        if args.out == "-":
            out_fp = sys.stdout
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            out_fp = open(args.out, "w", newline="", encoding="utf-8")
            close_after = True

        if args.format == "csv":
            write_csv(samples_iter, out_fp)
        else:
            write_jsonl(samples_iter, out_fp)
    finally:
        if close_after and out_fp is not None:
            out_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

