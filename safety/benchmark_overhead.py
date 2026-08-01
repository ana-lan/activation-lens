import torch
import time
import numpy as np
from scipy.stats import t as student_t
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

_clf_data = torch.load("results/multi_feature_classifier.pt")
_live_calib = torch.load("results/live_multifeature_calibration.pt")

SELECTED_FEATURES = torch.tensor(_clf_data["selected_features"], dtype=torch.long)
COEFFICIENTS = torch.tensor(_clf_data["coefficients"], dtype=torch.float32)
INTERCEPT = float(_clf_data["intercept"])
SCALER_MEAN = torch.tensor(_clf_data["scaler_mean"], dtype=torch.float32)
SCALER_SCALE = torch.tensor(_clf_data["scaler_scale"], dtype=torch.float32)
MIN_PREFIX_TOKENS = _live_calib["min_prefix_tokens"]
LIVE_THRESHOLD = _live_calib["live_threshold"]

def generate_baseline(model, prompt: str, max_new_tokens: int = 30):
    """One forward pass per token, no monitoring."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    generated_count = 0
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        generated_count += 1
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    return tokens, generated_count

def generate_monitored_naive(model, sae, prompt: str, max_new_tokens: int = 30, layer: int = 6):
    """One forward pass per token via run_with_cache, PLUS a full SAE forward
    (all 8,192 encoder features + full decoder reconstruction) — wasteful,
    included specifically to show the cost of NOT optimizing the monitor."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"blocks.{layer}.hook_resid_post"
    generated_count = 0

    running_sum = torch.zeros(len(SELECTED_FEATURES))
    num_seen = 0

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            token_activation = cache[hook_name][0, -1]
            features, _ = sae(token_activation.unsqueeze(0))  # full 8192-dim encode + decode
            selected_values = features[0, SELECTED_FEATURES]

            running_sum += selected_values
            num_seen += 1
            running_mean = running_sum / num_seen
            scaled = (running_mean - SCALER_MEAN) / SCALER_SCALE
            logit = torch.dot(scaled, COEFFICIENTS) + INTERCEPT
            score = torch.sigmoid(logit).item()
            eligible = num_seen >= MIN_PREFIX_TOKENS
            _ = eligible and score >= LIVE_THRESHOLD

        generated_count += 1
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    return tokens, generated_count

def generate_monitored_optimized(model, sae, prompt: str, max_new_tokens: int = 30, layer: int = 6):
    """One forward pass per token, computing ONLY the 100 selected encoder
    features directly (no full 8192-feature encode, no decoder at all) —
    the monitor you'd actually want to deploy."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"blocks.{layer}.hook_resid_post"
    generated_count = 0

    selected_W_enc = sae.W_enc[SELECTED_FEATURES]  # [100, 768]
    selected_b_enc = sae.b_enc[SELECTED_FEATURES]  # [100]

    running_sum = torch.zeros(len(SELECTED_FEATURES))
    num_seen = 0

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            token_activation = cache[hook_name][0, -1]
            selected_values = torch.relu(token_activation @ selected_W_enc.T + selected_b_enc)

            running_sum += selected_values
            num_seen += 1
            running_mean = running_sum / num_seen
            scaled = (running_mean - SCALER_MEAN) / SCALER_SCALE
            logit = torch.dot(scaled, COEFFICIENTS) + INTERCEPT
            score = torch.sigmoid(logit).item()
            eligible = num_seen >= MIN_PREFIX_TOKENS
            _ = eligible and score >= LIVE_THRESHOLD

        generated_count += 1
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    return tokens, generated_count

def run_paired_benchmark(prompt: str = "The best way to deal with a difficult situation is", max_new_tokens: int = 30, num_trials: int = 10):
    if num_trials < 2:
        raise ValueError("num_trials must be at least 2.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    print(f"Torch threads: {torch.get_num_threads()} | device: cpu | layer: 6 | KV-cache: NOT used\n")

    conditions = {"baseline": generate_baseline, "naive_monitor": generate_monitored_naive, "optimized_monitor": generate_monitored_optimized}
    times = {name: [] for name in conditions}
    generated_counts = {name: [] for name in conditions}

    print("Warmup...")
    for name, fn in conditions.items():
        args = (model, prompt, max_new_tokens) if name == "baseline" else (model, sae, prompt, max_new_tokens)
        for _ in range(3):
            fn(*args)

    trial_orders = [
        ["baseline", "naive_monitor", "optimized_monitor"],
        ["optimized_monitor", "baseline", "naive_monitor"],
        ["naive_monitor", "optimized_monitor", "baseline"],
    ]

    print(f"Running {num_trials} paired, counterbalanced trials...\n")
    for trial in range(num_trials):
        order = trial_orders[trial % len(trial_orders)]
        trial_outputs = {}

        for name in order:
            fn = conditions[name]
            args = (model, prompt, max_new_tokens) if name == "baseline" else (model, sae, prompt, max_new_tokens)

            start = time.perf_counter()
            tokens, count = fn(*args)
            elapsed = time.perf_counter() - start

            times[name].append(elapsed)
            generated_counts[name].append(count)
            trial_outputs[name] = tokens

        assert torch.equal(trial_outputs["baseline"], trial_outputs["naive_monitor"])
        assert torch.equal(trial_outputs["baseline"], trial_outputs["optimized_monitor"])

        print(f"  Trial {trial+1}/{num_trials} done (order: {' → '.join(order)})")

    assert generated_counts["baseline"] == generated_counts["naive_monitor"] == generated_counts["optimized_monitor"], \
        "Generated token counts differ between conditions."

    print("\nSanity check passed: all three conditions generated identical text across every trial.")

    results = {}
    baseline_times = np.array(times["baseline"])

    for name in conditions:
        arr = np.array(times[name])
        counts = np.array(generated_counts[name])
        tok_per_sec = counts / arr

        results[name] = {
            "mean_time": float(arr.mean()),
            "median_time": float(np.median(arr)),
            "std_time": float(arr.std(ddof=1)),
            "mean_tok_per_sec": float(tok_per_sec.mean()),
            "generated_counts": counts.tolist(),
            "raw_times": arr.tolist(),
        }

        if name != "baseline":
            paired_overhead_pct = (arr - baseline_times) / baseline_times * 100
            n = len(paired_overhead_pct)
            sem = paired_overhead_pct.std(ddof=1) / np.sqrt(n)
            critical = student_t.ppf(0.975, df=n - 1)
            ci95 = critical * sem
            results[name]["paired_overhead_pct_mean"] = float(paired_overhead_pct.mean())
            results[name]["paired_overhead_pct_95ci"] = float(ci95)

    print(f"\n--- Results ({num_trials} paired trials, prompt token count: {model.to_tokens(prompt).shape[1]}) ---")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  mean={r['mean_time']:.4f}s | median={r['median_time']:.4f}s | std={r['std_time']:.4f}s | {r['mean_tok_per_sec']:.2f} tok/s")
        if "paired_overhead_pct_mean" in r:
            print(f"  overhead vs baseline: {r['paired_overhead_pct_mean']:+.2f}% (95% CI ± {r['paired_overhead_pct_95ci']:.2f}%)")

    metadata = {
        "prompt": prompt, "max_new_tokens": max_new_tokens, "num_trials": num_trials,
        "torch_threads": torch.get_num_threads(), "device": "cpu", "layer": 6,
        "uses_kv_cache": False, "prompt_token_count": int(model.to_tokens(prompt).shape[1]),
        "note": "This benchmark does NOT use KV-caching. Absolute speed and relative overhead may differ in a production KV-cached serving engine.",
        "monitor_timing": "One-token-delayed: first step scores the final prompt token; final generated token is not scored.",
    }

    torch.save({"metadata": metadata, "results": results}, "results/overhead_benchmark.pt")
    return results

if __name__ == "__main__":
    run_paired_benchmark()