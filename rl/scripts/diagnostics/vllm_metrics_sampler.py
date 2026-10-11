#!/usr/bin/env python3
"""Poll scalar Prometheus metrics from local vLLM replicas."""
from __future__ import annotations
import argparse
import json
import signal
import time
import urllib.request
from pathlib import Path

RUNNING = True
METRICS = {
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:time_to_first_token_seconds_count", "vllm:time_to_first_token_seconds_sum",
    "vllm:request_queue_time_seconds_count", "vllm:request_queue_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count", "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_count", "vllm:request_decode_time_seconds_sum",
    "vllm:request_success_total",
}

def stop(*_args):
    global RUNNING
    RUNNING = False

def parse(body: str):
    values = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(" ", 1)[0]
        if metric not in METRICS:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        values[metric] = values.get(metric, 0.0) + value
    return values

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args=parser.parse_args()
    output=Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        while RUNNING:
            for replica, url in enumerate(args.url):
                row={"timestamp_ns":time.time_ns(),"event":"vllm_metrics","replica":replica,"url":url}
                try:
                    with urllib.request.urlopen(url.rstrip("/")+"/metrics", timeout=2) as response:
                        row["metrics"]=parse(response.read().decode("utf-8", errors="replace"))
                except Exception as error:
                    row["error_type"]=type(error).__name__
                handle.write(json.dumps(row,separators=(",",":"))+"\n")
            time.sleep(args.interval)
if __name__ == "__main__":
    main()
