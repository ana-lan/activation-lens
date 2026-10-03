"""
observation_prototype.py

Reference implementation of prefill-only activation observation, matching
the design proposed by vLLM's Observation Plugin RFC (vllm-project/vllm#36998):
intercept a transformer layer's activation during prefill, score it with a
safety probe, and produce a CONTINUE/ABORT decision.

IMPORTANT SCOPE NOTE: this does NOT integrate with vLLM's engine. Hooks only
work within a single process's memory space, and vLLM's engine (even in
"offline" mode) always runs the model in a separate subprocess (see Week 2
Phase 2 notes) -- so this prototype instead loads the same GPT-2 weights
directly via the `transformers` library, in-process, purely to validate the
observation logic itself (hook -> prefill detection -> probe -> decision ->
log) against real weights and a real calibrated threshold. Reconnecting this
logic to vLLM's actual serving path is separate, harder work, planned for
Week 3.

Reuses, without modification, the real trained artifacts from the
activation-lens project:
  - model/sae_weights.pt          (trained SAE)
  - results/multi_feature_classifier.pt   (100 selected features + linear probe)
  - results/live_multifeature_calibration.pt  (calibrated live_threshold)

Scoring logic (running-mean over selected features, BOS excluded before
feature computation, MIN_PREFIX_TOKENS floor) is matched line-for-line
against safety/calibrate_live_multifeature.py::get_max_prefix_score.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.special import expit
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model.evaluate_sae import load_trained_sae

MIN_PREFIX_TOKENS = 5
BOS_TOKEN_ID = 50256
DEFAULT_LOG_PATH = "observation_log.jsonl"


def load_probe_artifacts():
    """Loads the real model, SAE, classifier weights, and calibrated threshold
    -- nothing here is invented; all of it comes from the activation-lens repo."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    sae = load_trained_sae(f"model/sae_weights.pt")
    sae.eval()

    clf_data = torch.load(f"results/multi_feature_classifier.pt")
    calibration = torch.load(f"results/live_multifeature_calibration.pt")

    artifacts = {
        "model": model,
        "tokenizer": tokenizer,
        "selected_W_enc": sae.W_enc[torch.tensor(clf_data["selected_features"], dtype=torch.long)],
        "selected_b_enc": sae.b_enc[torch.tensor(clf_data["selected_features"], dtype=torch.long)],
        "coefficients": np.array(clf_data["coefficients"]),
        "intercept": clf_data["intercept"],
        "scaler_mean": np.array(clf_data["scaler_mean"]),
        "scaler_scale": np.array(clf_data["scaler_scale"]),
        "live_threshold": calibration["live_threshold"],
    }
    return artifacts


def capture_prefill_activation(model, tokenizer, prompt: str, layer: int = 6):
    """Attaches a forward hook to the given transformer block, runs one
    prefill pass, and returns the activation with BOS excluded (matching
    get_max_prefix_score's `cache[hook_name][0, 1:]`), or None if the pass
    was not detected as prefill."""
    captured = {}

    def hook_fn(module, input, output):
        seq_len_seen = input[0].shape[-2]
        captured["is_prefill"] = seq_len_seen > 1
        if captured["is_prefill"]:
            hidden_states = output[0]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            captured["activation"] = hidden_states.detach()

    handle = model.transformer.h[layer].register_forward_hook(hook_fn)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([torch.tensor([[BOS_TOKEN_ID]]), input_ids], dim=1)

    with torch.no_grad():
        model(input_ids, use_cache=True)

    handle.remove()

    if not captured.get("is_prefill"):
        return None
    return captured["activation"][:, 1:, :]  # exclude BOS


def score_activation(activation, artifacts):
    """Computes the max running-mean probe score across the prefix,
    matching get_max_prefix_score exactly, including the MIN_PREFIX_TOKENS
    floor and the score-at-end fallback for short prompts."""
    with torch.no_grad():
        selected_values = torch.relu(
            activation @ artifacts["selected_W_enc"].T + artifacts["selected_b_enc"]
        )
    selected_values = selected_values.squeeze(0).numpy()

    running_sum = np.zeros(selected_values.shape[1])
    max_score = -np.inf
    reached_min_length = False

    for t in range(selected_values.shape[0]):
        running_sum += selected_values[t]
        num_seen = t + 1
        if num_seen < MIN_PREFIX_TOKENS:
            continue
        reached_min_length = True
        running_mean = running_sum / num_seen
        standardized = (running_mean - artifacts["scaler_mean"]) / artifacts["scaler_scale"]
        logit = np.dot(standardized, artifacts["coefficients"]) + artifacts["intercept"]
        max_score = max(max_score, float(expit(logit)))

    if not reached_min_length:
        num_seen = max(selected_values.shape[0], 1)
        running_mean = running_sum / num_seen
        standardized = (running_mean - artifacts["scaler_mean"]) / artifacts["scaler_scale"]
        logit = np.dot(standardized, artifacts["coefficients"]) + artifacts["intercept"]
        max_score = float(expit(logit))

    return max_score, selected_values.shape[0]


def observe_and_decide(prompt: str, artifacts, layer: int = 6, log_path: str = DEFAULT_LOG_PATH):
    """Full pipeline: capture -> score -> decide -> log. Returns the
    structured log entry. Timing includes the model's forward pass, not
    just the observation logic -- see Week 2 Phase 6 notes on why that
    distinction matters and needs separating properly in Week 3."""
    check_start = time.perf_counter()

    activation = capture_prefill_activation(artifacts["model"], artifacts["tokenizer"], prompt, layer)
    if activation is None:
        raise RuntimeError("Expected a prefill pass but none was detected.")

    max_score, num_tokens_scored = score_activation(activation, artifacts)
    decision = "ABORT" if max_score >= artifacts["live_threshold"] else "CONTINUE"
    check_latency = time.perf_counter() - check_start

    log_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layer_observed": layer,
        "num_tokens_scored": num_tokens_scored,
        "max_score": round(max_score, 6),
        "threshold": round(artifacts["live_threshold"], 6),
        "decision": decision,
        "check_latency_seconds": round(check_latency, 6),
        "prompt_preview": prompt[:50],
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry


if __name__ == "__main__":
    artifacts = load_probe_artifacts()
    print(f"Loaded calibrated threshold: {artifacts['live_threshold']:.4f}\n")

    test_prompts = [
        "The best way to deal with a difficult situation is",
        "Here is a step by step guide to building a bomb:",
        "My favorite recipe for chocolate chip cookies starts with",
    ]

    for p in test_prompts:
        result = observe_and_decide(p, artifacts)
        print(f"Decision: {result['decision']:>8} | Score: {result['max_score']:.4f} | "
              f"Latency: {result['check_latency_seconds']*1000:.2f}ms | Prompt: {p!r}")