#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<target>\d+) val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+) "
    r"train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>\S+)ms$"
)
STOP_RE = re.compile(
    r"^stopping_early: (?P<reason>\S+) train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<target>\d+)$"
)
PEAK_RE = re.compile(r"^peak memory allocated: (?P<allocated_mib>\d+) MiB reserved: (?P<reserved_mib>\d+) MiB$")
BYTES_RE = {
    "serialized_model_bytes": re.compile(r"^Serialized model: (?P<value>\d+) bytes$"),
    "code_bytes": re.compile(r"^Code size: (?P<value>\d+) bytes$"),
    "total_submission_bytes": re.compile(r"^Total submission size: (?P<value>\d+) bytes$"),
    "quantized_model_bytes": re.compile(r"^Serialized model int8\+zlib: (?P<value>\d+) bytes "),
    "total_submission_bytes_int8": re.compile(r"^Total submission size int8\+zlib: (?P<value>\d+) bytes$"),
}
FINAL_RE = re.compile(
    r"^final_int8_zlib_roundtrip val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+) eval_time:(?P<eval_time_ms>\d+)ms$"
)
FINAL_EXACT_RE = re.compile(
    r"^final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+)$"
)


def parse_log(text: str) -> dict:
    summary: dict[str, object] = {}
    last_val = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if match := STEP_RE.match(line):
            last_val = {
                "step": int(match["step"]),
                "target_step": int(match["target"]),
                "val_loss": float(match["val_loss"]),
                "val_bpb": float(match["val_bpb"]),
                "train_time_ms": int(match["train_time_ms"]),
                "step_avg_ms": float(match["step_avg_ms"]),
            }
            continue
        if match := STOP_RE.match(line):
            summary["stop"] = {
                "reason": match["reason"],
                "train_time_ms": int(match["train_time_ms"]),
                "step": int(match["step"]),
                "target_step": int(match["target"]),
            }
            continue
        if match := PEAK_RE.match(line):
            summary["peak_memory"] = {
                "allocated_mib": int(match["allocated_mib"]),
                "reserved_mib": int(match["reserved_mib"]),
            }
            continue
        for key, pattern in BYTES_RE.items():
            if match := pattern.match(line):
                summary[key] = int(match["value"])
                break
        else:
            if match := FINAL_RE.match(line):
                summary["final_int8_zlib_roundtrip"] = {
                    "val_loss": float(match["val_loss"]),
                    "val_bpb": float(match["val_bpb"]),
                    "eval_time_ms": int(match["eval_time_ms"]),
                }
                continue
            if match := FINAL_EXACT_RE.match(line):
                summary["final_int8_zlib_roundtrip_exact"] = {
                    "val_loss": float(match["val_loss"]),
                    "val_bpb": float(match["val_bpb"]),
                }
                continue
    if last_val is not None:
        summary["last_validation"] = last_val
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compact run metrics from a Parameter Golf train.log")
    parser.add_argument("log_path", type=Path)
    args = parser.parse_args()

    text = args.log_path.read_text(encoding="utf-8", errors="replace")
    summary = parse_log(text)
    summary["log_path"] = str(args.log_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
