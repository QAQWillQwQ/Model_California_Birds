#!/usr/bin/env python3

import math
import re
import sys
import time
from pathlib import Path


EPOCH_HEADER_RE = re.compile(r"^===== Epoch (\d+)/(\d+) \(LR: ([0-9.eE+-]+)\) =====$")
BATCH_RE = re.compile(
    r"^\s*(Train|Val) Batch \[(\d+)/(\d+)\] \| Loss: ([0-9.]+) \| Top1: ([0-9.]+) \| Top5: ([0-9.]+)$"
)
EPOCH_SUMMARY_RE = re.compile(
    r"^Epoch \[(\d+)/(\d+)\] \| Train Loss: ([0-9.]+) \| Train Top1: ([0-9.]+) \| Train Top5: ([0-9.]+) "
    r"\| Val Loss: ([0-9.nan-]+) \| Val Top1: ([0-9.nan-]+) \| Val Top5: ([0-9.nan-]+) "
    r"\| LR: ([0-9.eE+-]+) \| Train Time: ([0-9.]+)s \| Val Time: ([0-9.]+)s \| Epoch Time: ([0-9.]+)s$"
)
BEST_CHECKPOINT_RE = re.compile(r"^Saved best checkpoint to:")


def usage() -> None:
    print("Usage: python scripts/monitor_training.py <debug_log_path> [poll_seconds]")


def parse_metric(value: str) -> float:
    if value == "nan":
        return math.nan
    return float(value)


def summarize_epoch(epoch_data: dict, best_val_top1: float, prior_epoch: dict | None, stagnation_count: int) -> list[str]:
    epoch_num = epoch_data["epoch"]
    val_top1 = epoch_data["val_top1"]
    val_loss = epoch_data["val_loss"]
    lr = epoch_data["lr"]

    lines = [
        (
            f"[epoch {epoch_num}] "
            f"train_top1={epoch_data['train_top1']:.4f} "
            f"val_top1={val_top1:.4f} "
            f"val_loss={val_loss:.4f} "
            f"lr={lr:.2e}"
        )
    ]

    if val_top1 >= best_val_top1:
        lines.append("status: new best validation so far")
    else:
        drop = best_val_top1 - val_top1
        lines.append(f"status: below best validation by {drop:.4f}")

    warnings = []
    if prior_epoch is not None:
        if val_top1 < prior_epoch["val_top1"]:
            warnings.append(f"val_top1 dropped by {prior_epoch['val_top1'] - val_top1:.4f} from previous epoch")
        if val_loss > prior_epoch["val_loss"]:
            warnings.append(f"val_loss rose by {val_loss - prior_epoch['val_loss']:.4f} from previous epoch")
        if epoch_data["train_top1"] < prior_epoch["train_top1"]:
            warnings.append(
                f"train_top1 dropped by {prior_epoch['train_top1'] - epoch_data['train_top1']:.4f} from previous epoch"
            )
        if lr > prior_epoch["lr"] and val_top1 < prior_epoch["val_top1"]:
            warnings.append("higher LR is coinciding with weaker validation; schedule may be too aggressive")

    if stagnation_count >= 3:
        warnings.append(f"no new best for {stagnation_count} epochs")

    if warnings:
        lines.append("watch: " + "; ".join(warnings))
    else:
        lines.append("watch: no obvious issue in this epoch")

    return lines


def monitor(debug_log_path: Path, poll_seconds: float) -> int:
    if not debug_log_path.is_file():
        print(f"Log not found: {debug_log_path}")
        return 1

    print(f"Monitoring: {debug_log_path}")
    print(f"Poll interval: {poll_seconds:.1f}s")

    last_offset = 0
    best_val_top1 = float("-inf")
    prior_epoch = None
    seen_epochs: set[int] = set()
    current_epoch = None
    last_progress = time.time()
    stagnation_count = 0

    while True:
        current_size = debug_log_path.stat().st_size
        if current_size < last_offset:
            print("Log file was truncated; restarting from beginning")
            last_offset = 0

        if current_size > last_offset:
            with debug_log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(last_offset)
                for raw_line in handle:
                    line = raw_line.rstrip("\n")

                    header_match = EPOCH_HEADER_RE.match(line)
                    if header_match:
                        current_epoch = int(header_match.group(1))
                        total_epochs = int(header_match.group(2))
                        lr = float(header_match.group(3))
                        last_progress = time.time()
                        print(f"[epoch {current_epoch}/{total_epochs}] started at lr={lr:.2e}")
                        continue

                    batch_match = BATCH_RE.match(line)
                    if batch_match:
                        phase = batch_match.group(1).lower()
                        batch_idx = int(batch_match.group(2))
                        total_batches = int(batch_match.group(3))
                        loss = float(batch_match.group(4))
                        top1 = float(batch_match.group(5))
                        top5 = float(batch_match.group(6))
                        last_progress = time.time()
                        if batch_idx == total_batches or batch_idx % 100 == 0:
                            print(
                                f"[{phase}] epoch={current_epoch} batch={batch_idx}/{total_batches} "
                                f"loss={loss:.4f} top1={top1:.4f} top5={top5:.4f}"
                            )
                        continue

                    if BEST_CHECKPOINT_RE.match(line):
                        print("status: best checkpoint updated")
                        continue

                    summary_match = EPOCH_SUMMARY_RE.match(line)
                    if summary_match:
                        epoch_num = int(summary_match.group(1))
                        if epoch_num in seen_epochs:
                            continue

                        seen_epochs.add(epoch_num)
                        epoch_data = {
                            "epoch": epoch_num,
                            "train_loss": float(summary_match.group(3)),
                            "train_top1": float(summary_match.group(4)),
                            "train_top5": float(summary_match.group(5)),
                            "val_loss": parse_metric(summary_match.group(6)),
                            "val_top1": parse_metric(summary_match.group(7)),
                            "val_top5": parse_metric(summary_match.group(8)),
                            "lr": float(summary_match.group(9)),
                            "train_time": float(summary_match.group(10)),
                            "val_time": float(summary_match.group(11)),
                            "epoch_time": float(summary_match.group(12)),
                        }

                        if epoch_data["val_top1"] > best_val_top1:
                            best_val_top1 = epoch_data["val_top1"]
                            stagnation_count = 0
                        else:
                            stagnation_count += 1

                        for message in summarize_epoch(epoch_data, best_val_top1, prior_epoch, stagnation_count):
                            print(message)

                        prior_epoch = epoch_data
                        last_progress = time.time()

                last_offset = handle.tell()

        idle_seconds = time.time() - last_progress
        if idle_seconds > max(120.0, poll_seconds * 4):
            print(f"watch: no new log activity for {idle_seconds:.0f}s")
            last_progress = time.time()

        time.sleep(poll_seconds)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        usage()
        return 1

    debug_log_path = Path(sys.argv[1]).expanduser().resolve()
    poll_seconds = 15.0
    if len(sys.argv) == 3:
        poll_seconds = float(sys.argv[2])
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

    return monitor(debug_log_path, poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped monitor")
        raise SystemExit(130)
