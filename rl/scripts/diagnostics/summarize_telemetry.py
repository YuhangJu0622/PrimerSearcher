#!/usr/bin/env python3
"""Summarize content-free OpenSeeker RL diagnostic JSONL files."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * pct / 100
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def load_jsonl(paths):
    rows = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def values_summary(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": mean(values), "p50": percentile(values, 50), "p95": percentile(values, 95), "max": max(values), "sum": sum(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Telemetry directory under an experiment output")
    parser.add_argument("--output", help="Write JSON report here; default: <dir>/summary.json")
    args = parser.parse_args()
    root = Path(args.dir)
    agent_rows = load_jsonl(root.glob("agent/*.jsonl"))
    phase_rows = load_jsonl(root.glob("phase/*.jsonl"))
    resource_rows = load_jsonl([root / "resources.jsonl"] if (root / "resources.jsonl").exists() else [])

    by_request = defaultdict(list)
    for row in agent_rows:
        by_request[row.get("trace_id")].append(row)
    trajectories = [row for row in agent_rows if row.get("event") == "trajectory_complete"]
    vllm_complete = [row for row in agent_rows if row.get("event") == "vllm_complete"]
    vllm_error = [row for row in agent_rows if row.get("event") == "vllm_error"]
    tools = [row for row in agent_rows if row.get("event") == "tool_complete"]

    intervals = []
    for trace_rows in by_request.values():
        starts = {(r.get("turn"), r.get("attempt")): r for r in trace_rows if r.get("event") == "vllm_submit"}
        for finish in trace_rows:
            key = (finish.get("turn"), finish.get("attempt"))
            if finish.get("event") == "vllm_complete" and key in starts:
                intervals.append((starts[key]["timestamp_ns"], finish["timestamp_ns"]))
    marks = []
    for start, finish in intervals:
        marks.append((start, 1)); marks.append((finish, -1))
    active = peak_active = 0
    for _ts, delta in sorted(marks):
        active += delta; peak_active = max(peak_active, active)

    phase = defaultdict(list)
    for row in phase_rows:
        if row.get("event") == "phase_complete":
            phase[row.get("phase")].append(row.get("elapsed_s"))

    gpu = defaultdict(lambda: defaultdict(list))
    cgroup_memory = []
    nvml_available = 0
    for row in resource_rows:
        if row.get("cgroup_memory_current_bytes") is not None:
            cgroup_memory.append(row["cgroup_memory_current_bytes"])
        nvml_available += bool(row.get("nvml_available"))
        for device in row.get("gpus", []):
            index = str(device.get("index"))
            for field in ("gpu_util_percent", "memory_util_percent", "memory_used_bytes", "pcie_tx_kib_s", "pcie_rx_kib_s"):
                if device.get(field) is not None:
                    gpu[index][field].append(device[field])

    report = {
        "telemetry_dir": str(root),
        "trajectories": {
            "completed": len(trajectories),
            "elapsed_s": values_summary([r.get("elapsed_s") for r in trajectories if r.get("elapsed_s") is not None]),
            "response_tokens": values_summary([r.get("response_tokens") for r in trajectories if r.get("response_tokens") is not None]),
            "assistant_turns": values_summary([r.get("total_assistant_turns") for r in trajectories if r.get("total_assistant_turns") is not None]),
            "termination_reasons": dict(Counter(r.get("termination_reason") for r in trajectories)),
        },
        "vllm": {
            "requests_completed": len(vllm_complete),
            "requests_failed": len(vllm_error),
            "client_elapsed_s": values_summary([r.get("elapsed_s") for r in vllm_complete if r.get("elapsed_s") is not None]),
            "server_elapsed_s": values_summary([r.get("server_elapsed_s") for r in vllm_complete if r.get("server_elapsed_s") is not None]),
            "first_output_s": values_summary([r.get("server_first_output_s") for r in vllm_complete if r.get("server_first_output_s") is not None]),
            "client_minus_server_s": values_summary([r["elapsed_s"] - r["server_elapsed_s"] for r in vllm_complete if r.get("elapsed_s") is not None and r.get("server_elapsed_s") is not None]),
            "generated_tokens": values_summary([r.get("generated_tokens") for r in vllm_complete if r.get("generated_tokens") is not None]),
            "requests_by_replica": dict(Counter(str(r.get("replica_rank")) for r in vllm_complete)),
            "max_concurrent_agent_requests": peak_active,
        },
        "tools": {
            "completed": len(tools),
            "failures": sum(not bool(r.get("success")) for r in tools),
            "cache_hits": sum(bool(r.get("cached")) for r in tools),
            "elapsed_s": values_summary([r.get("elapsed_s") for r in tools if r.get("elapsed_s") is not None]),
            "by_tool": dict(Counter(r.get("tool") for r in tools)),
        },
        "weight_sync_phases_s": {name: values_summary([value for value in durations if value is not None]) for name, durations in phase.items()},
        "resources": {
            "samples": len(resource_rows),
            "samples_with_nvml": nvml_available,
            "cgroup_memory_current_bytes": values_summary(cgroup_memory),
            "gpus": {index: {field: values_summary(values) for field, values in fields.items()} for index, fields in gpu.items()},
        },
    }
    output = Path(args.output) if args.output else root / "summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
