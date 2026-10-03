"""
benchmarks/vllm_harness.py

Reusable, config-driven benchmark harness for a running vLLM OpenAI-compatible
server. Measures throughput, TTFT, inter-token latency, P50/P95/P99 latency,
and peak GPU memory, across configurable concurrency levels.

Built and validated in Week 1 (Phases 1-7): streaming-based per-request timing,
a sequential baseline, a concurrency sweep, GPU memory tracking via pynvml,
YAML-driven configuration, and a repeated-run reliability check.

Requires a vLLM server already running and reachable at the URL given in the
config (see configs/concurrency_sweep.yaml for an example). Does not start or
stop the server itself -- that stays the caller's responsibility, matching how
Week 1's Colab notebook launched it via subprocess.

Environment this was validated against (see /profile or README for updates):
  vLLM 0.27.1, torch 2.13.0+cu130, Python 3.12.13, NVIDIA L4
"""

import json
import threading
import time
from datetime import datetime, timezone

import numpy as np
import requests
import yaml

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Single-request timing (Phase 2)
# ---------------------------------------------------------------------------

def send_streaming_request(prompt, max_tokens=30, url="http://localhost:8000/v1/completions"):
    """Sends one streaming completion request and times it precisely:
    TTFT (time to first token) separately from total latency and the
    average gap between subsequent tokens."""
    payload = {
        "model": "gpt2",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_timestamps = []
    generated_text = ""

    with requests.post(url, json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                chunk = json.loads(data_str)
                token_text = chunk["choices"][0]["text"]
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_timestamps.append(now)
                generated_text += token_text

    end_time = time.perf_counter()
    ttft = first_token_time - start_time if first_token_time else None
    total_latency = end_time - start_time
    gaps = [token_timestamps[i] - token_timestamps[i - 1] for i in range(1, len(token_timestamps))]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    return {
        "prompt": prompt,
        "ttft": ttft,
        "total_latency": total_latency,
        "avg_inter_token": avg_gap,
        "num_tokens": len(token_timestamps),
    }


# ---------------------------------------------------------------------------
# GPU memory tracking (Phase 5)
# ---------------------------------------------------------------------------

class GPUMemoryMonitor:
    """Samples GPU memory (across ALL processes, via NVML) in a background
    thread while a benchmark runs. Use around code that hits a vLLM server
    running in a separate process -- torch.cuda memory calls from this
    process would not see the server's usage."""

    def __init__(self, interval=0.05, gpu_index=0):
        if not _NVML_AVAILABLE:
            raise RuntimeError("pynvml is required for GPU memory tracking: pip install pynvml")
        self.interval = interval
        self.readings = []
        self._stop_event = threading.Event()
        self._thread = None
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def _get_memory_used_mb(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used / (1024 ** 2)

    def _run(self):
        while not self._stop_event.is_set():
            self.readings.append(self._get_memory_used_mb())
            time.sleep(self.interval)

    def start(self):
        self.readings = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        return {
            "peak_mb": max(self.readings) if self.readings else None,
            "avg_mb": sum(self.readings) / len(self.readings) if self.readings else None,
            "num_samples": len(self.readings),
        }


# ---------------------------------------------------------------------------
# Concurrency sweep (Phase 3-4)
# ---------------------------------------------------------------------------

import concurrent.futures


def run_concurrent_batch(prompts, concurrency, max_tokens=30, url="http://localhost:8000/v1/completions"):
    """Sends `prompts` with up to `concurrency` requests in flight at once.
    At concurrency=1 this reduces to Week 1's sequential baseline."""
    results = []
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_streaming_request, p, max_tokens, url) for p in prompts]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    end = time.perf_counter()
    wall_time = end - start

    total_tokens = sum(r["num_tokens"] for r in results)
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    latencies = [r["total_latency"] for r in results]

    return {
        "concurrency": concurrency,
        "num_requests": len(prompts),
        "wall_time": wall_time,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "avg_ttft": float(np.mean(ttfts)) if ttfts else None,
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "p99_latency": float(np.percentile(latencies, 99)),
    }


# ---------------------------------------------------------------------------
# Config-driven runs (Phase 6)
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_benchmark_from_config(config_path, results_dir="results"):
    """Runs a full concurrency sweep as described by a YAML config, tracking
    GPU memory per condition, and saves a self-documenting JSON result file
    (config + environment info + timestamp + results, all in one place)."""
    config = load_config(config_path)
    bench = config["benchmark"]

    url = config["server"]["url"]
    max_tokens = bench["max_tokens"]
    prompts = bench["prompts"] * bench["repeats_per_prompt"]

    print(f"Running test: {config['test_name']}")
    print("Warming up...")
    for _ in range(bench["warmup_requests"]):
        send_streaming_request("This is a warm-up request.", max_tokens, url)
    print("Warm-up done.\n")

    sweep_results = []
    for c in bench["concurrency_levels"]:
        if _NVML_AVAILABLE:
            monitor = GPUMemoryMonitor(interval=0.05)
            monitor.start()

        r = run_concurrent_batch(prompts, concurrency=c, max_tokens=max_tokens, url=url)

        if _NVML_AVAILABLE:
            mem_stats = monitor.stop()
            r["peak_memory_mb"] = mem_stats["peak_mb"]

        sweep_results.append(r)

        mem_str = f" | Peak GPU mem: {r.get('peak_memory_mb', 'n/a'):.1f} MB" if r.get("peak_memory_mb") else ""
        print(f"Concurrency={c:>2} | Throughput: {r['throughput']:.2f} tok/s | "
              f"P99: {r['p99_latency']:.4f}s{mem_str}")

    output = {
        "test_name": config["test_name"],
        "config_used": config,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": sweep_results,
    }

    import os
    os.makedirs(results_dir, exist_ok=True)
    out_path = f"{results_dir}/{config['test_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved results to: {out_path}")
    return output


def check_reliability(config_path, num_runs=3, results_dir="results"):
    """Runs the same config multiple times and reports the coefficient of
    variation (std dev as % of mean) per concurrency level, so a single
    future run can be judged trustworthy or not against known noise."""
    all_runs = []
    for i in range(num_runs):
        print(f"\n=== Reliability run {i + 1}/{num_runs} ===")
        result = run_benchmark_from_config(config_path, results_dir=results_dir)
        all_runs.append(result)

    concurrency_levels = [r["concurrency"] for r in all_runs[0]["results"]]

    print("\n=== Reliability summary ===")
    summary = []
    for idx, c in enumerate(concurrency_levels):
        throughputs = [run["results"][idx]["throughput"] for run in all_runs]
        mean_tp = float(np.mean(throughputs))
        std_tp = float(np.std(throughputs))
        cv = (std_tp / mean_tp) * 100 if mean_tp else 0

        print(f"Concurrency={c:>2} | Throughputs: {[round(t, 1) for t in throughputs]} | "
              f"Mean: {mean_tp:.1f} | Std dev: {std_tp:.1f} | CV: {cv:.1f}%")
        summary.append({"concurrency": c, "throughputs": throughputs, "mean": mean_tp, "std": std_tp, "cv_pct": cv})

    return all_runs, summary


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/configs/concurrency_sweep.yaml"
    run_benchmark_from_config(config_path)
