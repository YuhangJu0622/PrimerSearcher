#!/usr/bin/env python3
"""One-second, content-free host/GPU sampler for RL performance diagnosis."""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import psutil
import pynvml

RUNNING = True
NVML_AVAILABLE = False
NVML_ERROR: str | None = None

def stop(*_args):
    global RUNNING
    RUNNING = False

def read_int(path: str):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None

def process_stats(pid: int):
    try:
        process = psutil.Process(pid)
        info = process.memory_full_info()
        return {
            "pid": pid,
            "name": process.name(),
            "rss_bytes": info.rss,
            "pss_bytes": getattr(info, "pss", None),
            "cpu_percent": process.cpu_percent(interval=None),
        }
    except (psutil.Error, OSError):
        return {"pid": pid}

def gpu_snapshot():
    if not NVML_AVAILABLE:
        return []
    gpus = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
        except AttributeError:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError:
            processes = []
        gpu_processes = []
        for process in processes:
            entry = process_stats(process.pid)
            unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
            used_gpu_memory = getattr(process, "usedGpuMemory", None)
            entry["gpu_memory_bytes"] = None if unavailable is not None and used_gpu_memory == unavailable else used_gpu_memory
            gpu_processes.append(entry)
        def optional(call):
            try:
                return call()
            except pynvml.NVMLError:
                return None
        gpus.append({
            "index": index,
            "gpu_util_percent": utilization.gpu,
            "memory_util_percent": utilization.memory,
            "memory_used_bytes": memory.used,
            "memory_total_bytes": memory.total,
            "power_mw": optional(lambda: pynvml.nvmlDeviceGetPowerUsage(handle)),
            "temperature_c": optional(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
            "pcie_tx_kib_s": optional(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)),
            "pcie_rx_kib_s": optional(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)),
            "compute_processes": gpu_processes,
        })
    return gpus

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    global NVML_AVAILABLE, NVML_ERROR
    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except pynvml.NVMLError as error:
        NVML_ERROR = f"{type(error).__name__}: {error}"
    try:
        with output.open("a", encoding="utf-8", buffering=1) as handle:
            while RUNNING:
                now = time.time_ns()
                vm = psutil.virtual_memory()
                record = {
                    "timestamp_ns": now,
                    "event": "resource_sample",
                    "host_memory_used_bytes": vm.used,
                    "host_memory_available_bytes": vm.available,
                    "cgroup_memory_current_bytes": read_int("/sys/fs/cgroup/memory.current"),
                    "cgroup_memory_max_bytes": read_int("/sys/fs/cgroup/memory.max"),
                    "cgroup_memory_swap_current_bytes": read_int("/sys/fs/cgroup/memory.swap.current"),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "nvml_available": NVML_AVAILABLE,
                    "nvml_error": NVML_ERROR,
                    "gpus": gpu_snapshot(),
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                time.sleep(args.interval)
    finally:
        if NVML_AVAILABLE:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
