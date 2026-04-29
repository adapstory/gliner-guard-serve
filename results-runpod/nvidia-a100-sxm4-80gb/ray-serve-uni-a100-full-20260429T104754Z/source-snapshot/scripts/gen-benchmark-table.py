#!/usr/bin/env python3
"""Parse results/*.csv (Locust stats) and print a markdown benchmark table."""

import csv
import glob
import os
import sys


def parse_csv(path: str) -> dict | None:
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] == "Aggregated":
                return {
                    "rps": float(row["Requests/s"]),
                    "p50": int(row["50%"]),
                    "p95": int(row["95%"]),
                }
    return None


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not files:
        print("No CSV files found in results/", file=sys.stderr)
        sys.exit(1)

    lines = [
        "| benchmark | RPS | P50 (ms) | P95 (ms) |",
        "|-----------|----:|--------:|---------:|",
    ]
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        stats = parse_csv(path)
        if stats is None:
            continue
        lines.append(
            f"| {name} | {stats['rps']:.1f} | {stats['p50']} | {stats['p95']} |"
        )

    print("\n".join(lines))


if __name__ == "__main__":
    main()