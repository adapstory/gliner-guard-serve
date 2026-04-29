#!/usr/bin/env python3
"""Plot Locust full-history throughput and latency with mean +/- std bands."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True, help="Glob for *_stats_history.csv")
    parser.add_argument("--output", required=True, help="PNG output path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--latency-percentile", default="95%", choices=["50%", "95%"])
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--rolling-window", type=int, default=3)
    parser.add_argument(
        "--throughput-scale",
        type=float,
        default=1.0,
        help="Scale throughput for same-axis plots, e.g. 1000 for rps*1000.",
    )
    return parser.parse_args()


def load_history(pattern: str, latency_percentile: str, max_users: int | None) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No history files match: {pattern}")

    frames = []
    for run_id, path in enumerate(files, start=1):
        df = pd.read_csv(path)
        df = df[
            (df["Name"] == "Aggregated")
            & (df["User Count"] > 0)
            & (df["Requests/s"] > 0)
            & (df[latency_percentile] > 0)
        ].copy()
        if max_users is not None:
            df = df[df["User Count"] <= max_users]
        df["Run"] = run_id
        df["Throughput"] = df["Requests/s"]
        df["Latency"] = df[latency_percentile]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def aggregate(raw: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    agg = (
        raw.groupby("User Count")
        .agg(
            Throughput_mean=("Throughput", "mean"),
            Throughput_std=("Throughput", "std"),
            Latency_mean=("Latency", "mean"),
            Latency_std=("Latency", "std"),
            Runs=("Run", "nunique"),
        )
        .reset_index()
        .sort_values("User Count")
    )
    agg[["Throughput_std", "Latency_std"]] = agg[["Throughput_std", "Latency_std"]].fillna(0)
    for col in ["Throughput_mean", "Latency_mean"]:
        agg[col + "_smooth"] = (
            agg[col].rolling(rolling_window, center=True, min_periods=1).mean()
        )
    agg["score_mean"] = agg["Throughput_mean"] / agg["Latency_mean"]
    return agg


def plot(agg: pd.DataFrame, output: str, title: str | None, latency_percentile: str, throughput_scale: float) -> None:
    sns.set_theme(style="darkgrid")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best = agg.loc[agg["score_mean"].idxmax()]
    optimal_users = int(best["User Count"])
    best_thr = float(best["Throughput_mean"])
    best_lat = float(best["Latency_mean"])
    runs = int(agg["Runs"].max())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = agg["User Count"].to_numpy()

    thr = (agg["Throughput_mean_smooth"] * throughput_scale).to_numpy()
    thr_std = (agg["Throughput_std"] * throughput_scale).to_numpy()
    lat = agg["Latency_mean_smooth"].to_numpy()
    lat_std = agg["Latency_std"].to_numpy()

    ax.plot(x, thr, linewidth=2.2, color="blue", label=f"Throughput avg+/-std, rps*{throughput_scale:g}")
    ax.fill_between(x, thr - thr_std, thr + thr_std, color="blue", alpha=0.25)
    ax.plot(x, lat, linewidth=2.0, linestyle="--", color="orange", label=f"Latency {latency_percentile} avg+/-std, ms")
    ax.fill_between(x, lat - lat_std, lat + lat_std, color="orange", alpha=0.25)
    ax.axvline(
        optimal_users,
        color="black",
        linestyle="--",
        linewidth=2.2,
        label=f"Optimal:\nLatency: {best_lat:.2f} ms\nThroughput: {best_thr:.2f} rps",
    )

    ax.set_xlabel("Num users")
    ax.set_ylabel("Value, ms")
    ax.set_title(title or f"Throughput & {latency_percentile} Latency with Confidence Intervals, N_experiments = {runs}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Wrote {output_path}")
    print(f"Optimal users={optimal_users}, throughput={best_thr:.2f} rps, latency={best_lat:.2f} ms")


def main() -> None:
    args = parse_args()
    raw = load_history(args.glob, args.latency_percentile, args.max_users)
    agg = aggregate(raw, args.rolling_window)
    plot(agg, args.output, args.title, args.latency_percentile, args.throughput_scale)


if __name__ == "__main__":
    main()
