#!/usr/bin/env python3
"""Monitor a bare-metal A100 Ray Serve benchmark result directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ERROR_PATTERNS = [
    "Traceback",
    "ModuleNotFoundError",
    "ActorDiedError",
    "RuntimeError:",
    "command not found",
    "Stale file handle",
    "No space left on device",
    "CUDA out of memory",
    "OutOfMemoryError",
    "TIMEOUT waiting",
]


@dataclass
class LogSnippet:
    path: str
    line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--driver-log", default=None)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--expected-runs", type=int, default=18)
    parser.add_argument("--stale-minutes", type=int, default=25)
    parser.add_argument("--write-status", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_text_tail(path: Path, max_bytes: int = 128_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        return fh.read().decode("utf-8", errors="replace")


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(result_dir: Path, pid: int | None, pid_file: str | None) -> int | None:
    if pid:
        return pid
    candidates = []
    if pid_file:
        candidates.append(Path(pid_file))
    candidates.append(Path(str(result_dir) + ".pid"))
    candidates.append(result_dir / "monitor" / "experiment.pid")
    for candidate in candidates:
        if candidate.exists():
            raw = candidate.read_text().strip()
            if raw.isdigit():
                return int(raw)
    return None


def read_summary(result_dir: Path) -> list[dict[str, str]]:
    summary = result_dir / "summary.csv"
    if not summary.exists():
        return []
    with summary.open(newline="") as fh:
        return list(csv.DictReader(fh))


def latest_file(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def detect_active_prefix(driver_text: str, result_dir: Path) -> str | None:
    matches = re.findall(r"Ray Serve (?:rest|grpc): ([^\n]+)", driver_text)
    if matches:
        return matches[-1].strip()

    logs_dir = result_dir / "logs"
    latest_locust = latest_file(list(logs_dir.glob("*-locust.log")))
    if latest_locust:
        return latest_locust.name.removesuffix("-locust.log")
    return None


def parse_latest_aggregated(locust_log: Path | None) -> str | None:
    if not locust_log or not locust_log.exists():
        return None
    lines = read_text_tail(locust_log, 64_000).splitlines()
    aggregated = [line.strip() for line in lines if line.strip().startswith("Aggregated")]
    return aggregated[-1] if aggregated else None


def find_recent_errors(result_dir: Path, driver_log: Path) -> list[LogSnippet]:
    candidates = [driver_log]
    logs_dir = result_dir / "logs"
    candidates.extend(sorted(logs_dir.glob("*.log")))

    snippets: list[LogSnippet] = []
    for path in candidates:
        if not path.exists():
            continue
        for line in read_text_tail(path, 96_000).splitlines():
            if "Aborted" in line and "core dumped" in line:
                continue
            if any(pattern in line for pattern in ERROR_PATTERNS):
                snippets.append(LogSnippet(str(path), line.strip()))
    return snippets[-12:]


def latest_gpu_sample(result_dir: Path, active_prefix: str | None) -> str | None:
    candidates = []
    if active_prefix:
        candidates.append(result_dir / f"gpu-{active_prefix}.csv")
    candidates.extend(sorted(result_dir.glob("gpu-*.csv")))
    gpu_file = latest_file(candidates)
    if not gpu_file:
        return None
    lines = [line.strip() for line in read_text_tail(gpu_file, 16_000).splitlines() if line.strip()]
    return lines[-1] if len(lines) > 1 else None


def build_status(args: argparse.Namespace) -> dict:
    result_dir = Path(args.result_dir)
    driver_log = Path(args.driver_log) if args.driver_log else Path(str(result_dir) + ".driver.log")
    pid = read_pid(result_dir, args.pid, args.pid_file)
    alive = process_alive(pid)
    summary_rows = read_summary(result_dir)
    completed_runs = len(summary_rows)
    driver_text = read_text_tail(driver_log)
    active_prefix = detect_active_prefix(driver_text, result_dir)
    active_locust_log = result_dir / "logs" / f"{active_prefix}-locust.log" if active_prefix else None
    latest_aggregated = parse_latest_aggregated(active_locust_log)
    latest_activity = 0.0
    activity_sources = [driver_log, result_dir / "summary.csv"]
    if active_locust_log:
        activity_sources.append(active_locust_log)
    for source in activity_sources:
        if source.exists():
            latest_activity = max(latest_activity, source.stat().st_mtime)
    stale_seconds = time.time() - latest_activity if latest_activity else None
    stale = stale_seconds is not None and stale_seconds > args.stale_minutes * 60
    errors = find_recent_errors(result_dir, driver_log)

    if completed_runs >= args.expected_runs:
        state = "completed"
    elif alive and stale:
        state = "stalled"
    elif alive:
        state = "running"
    else:
        state = "failed"

    last_completed = summary_rows[-1] if summary_rows else None
    remaining_runs = max(args.expected_runs - completed_runs, 0)

    return {
        "state": state,
        "checked_at_utc": utc_now().isoformat(timespec="seconds"),
        "result_dir": str(result_dir),
        "driver_log": str(driver_log),
        "pid": pid,
        "process_alive": alive,
        "expected_runs": args.expected_runs,
        "completed_runs": completed_runs,
        "remaining_runs": remaining_runs,
        "active_prefix": active_prefix,
        "active_locust_log": str(active_locust_log) if active_locust_log else None,
        "latest_activity_utc": datetime.fromtimestamp(latest_activity, timezone.utc).isoformat(timespec="seconds")
        if latest_activity
        else None,
        "stale_seconds": int(stale_seconds) if stale_seconds is not None else None,
        "latest_aggregated": latest_aggregated,
        "latest_gpu_sample": latest_gpu_sample(result_dir, active_prefix),
        "last_completed": last_completed,
        "recent_errors": [{"path": item.path, "line": item.line} for item in errors],
    }


def render_markdown(status: dict) -> str:
    lines = [
        f"# A100 Experiment Status: {status['state']}",
        "",
        f"- checked_at_utc: {status['checked_at_utc']}",
        f"- result_dir: `{status['result_dir']}`",
        f"- pid: `{status['pid']}` alive={status['process_alive']}",
        f"- progress: {status['completed_runs']}/{status['expected_runs']} completed, {status['remaining_runs']} remaining",
        f"- active: `{status['active_prefix']}`",
        f"- latest_activity_utc: {status['latest_activity_utc']}",
    ]
    if status.get("latest_aggregated"):
        lines.append(f"- latest_locust: `{status['latest_aggregated']}`")
    if status.get("latest_gpu_sample"):
        lines.append(f"- latest_gpu: `{status['latest_gpu_sample']}`")
    if status.get("last_completed"):
        completed = status["last_completed"]
        lines.append(
            "- last_completed: "
            f"`{completed.get('prefix')}` rps={completed.get('rps')} "
            f"p95={completed.get('p95_ms')} failures={completed.get('failures')}"
        )
    if status.get("recent_errors"):
        lines.extend(["", "## Recent Errors"])
        for item in status["recent_errors"]:
            lines.append(f"- `{item['path']}`: {item['line']}")
    return "\n".join(lines) + "\n"


def write_status(result_dir: Path, status: dict) -> None:
    monitor_dir = result_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    (monitor_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    (monitor_dir / "status.md").write_text(render_markdown(status))
    with (monitor_dir / "status-history.jsonl").open("a") as fh:
        fh.write(json.dumps(status, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    status = build_status(args)
    if args.write_status:
        write_status(Path(args.result_dir), status)
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(status))


if __name__ == "__main__":
    main()
