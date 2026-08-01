import torch
import numpy as np
from datasets import load_dataset
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

NEUTRAL_PROMPTS = [
    "My favorite thing to do on the weekend is",
    "Yesterday I went to the store and",
    "The most important skill in life is",
    "When I think about the future, I",
    "Here is a short story about a dog:",
]

def get_activation_statistics_multi(model, sae, feature_indices, layer=6, start=10000, num_texts=50):
    """Computes activation statistics for ALL candidate features in ONE
    pass over the documents — 50 model passes total instead of 50 per feature."""
    dataset = load_dataset("Skylion007/openwebtext", split=f"train[{start}:{start + num_texts}]")
    hook_name = f"blocks.{layer}.hook_resid_post"
    collected = {feature_idx: [] for feature_idx in feature_indices}

    for example in dataset:
        text = example["text"].strip()
        if not text:
            continue

        tokens = model.to_tokens(text, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name, return_type=None)
            activations = cache[hook_name][0, 1:]
            features, _ = sae(activations)

        for feature_idx in feature_indices:
            collected[feature_idx].append(features[:, feature_idx].cpu())

        del features, activations, cache

    stats = {}
    for feature_idx, chunks in collected.items():
        values = torch.cat(chunks)
        positive = values[values > 0]
        stats[feature_idx] = {
            "p99_all": torch.quantile(values, 0.99).item(),
            "p95_active": torch.quantile(positive, 0.95).item() if positive.numel() else 0.0,
            "prevalence": positive.numel() / values.numel(),
        }

    return stats

def get_steering_direction(sae, feature_idx):
    direction = sae.W_dec[:, feature_idx].clone()
    assert torch.isfinite(direction).all()
    norm = direction.norm()
    if norm.item() <= 1e-8:
        raise ValueError(f"Feature {feature_idx} has a zero-norm decoder direction.")
    return direction / norm, norm.item()

def generate_baseline(model, prompt, max_new_tokens=25):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    prompt_len = tokens.shape[1]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    completion_tokens = tokens[0, prompt_len:]
    return {"completion": model.to_string(completion_tokens), "completion_token_ids": completion_tokens.cpu().tolist()}

def generate_with_steering(model, sae, feature_idx, direction, strength, prompt, max_new_tokens=25, layer=6):
    hook_name = f"blocks.{layer}.hook_resid_post"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    prompt_len = tokens.shape[1]
    activation_trace = []

    encoder_direction = sae.W_enc[feature_idx]
    encoder_bias = sae.b_enc[feature_idx]

    def steering_hook(residual, hook):
        modified = residual.clone()
        modified[:, -1, :] += strength * direction
        with torch.no_grad():
            post_activation = torch.relu(modified[:, -1, :] @ encoder_direction + encoder_bias)
        activation_trace.append(post_activation.item())
        return modified

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, steering_hook)])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == model.tokenizer.eos_token_id:
            break

    completion_tokens = tokens[0, prompt_len:]
    return {
        "completion": model.to_string(completion_tokens),
        "completion_token_ids": completion_tokens.cpu().tolist(),
        "mean_post_activation": float(np.mean(activation_trace)),
        "max_post_activation": float(np.max(activation_trace)),
    }

def test_candidate(model, sae, feature_idx, stats, layer=6, strength_multiples=(0, 3, 5, 10)):
    direction, dec_norm = get_steering_direction(sae, feature_idx)

    activation_reference = stats["p99_all"]
    if activation_reference <= 1e-8:
        activation_reference = stats["p95_active"]
    if activation_reference <= 1e-8:
        print(f"\nFeature #{feature_idx}: never activated, skipping.")
        return

    print(f"\n{'='*70}")
    print(f"Feature #{feature_idx} | decoder_norm={dec_norm:.4f} | p99_all={stats['p99_all']:.4f} | prevalence={stats['prevalence']:.4f}")
    print(f"{'='*70}")

    for mult in strength_multiples:
        strength = mult * activation_reference * dec_norm
        print(f"\n  --- strength = {mult}x reference ({strength:.4f}) ---")
        for prompt in NEUTRAL_PROMPTS:
            if mult == 0:
                baseline = generate_baseline(model, prompt)
                result = generate_with_steering(model, sae, feature_idx, direction, 0.0, prompt)
                if result["completion_token_ids"] != baseline["completion_token_ids"]:
                    raise AssertionError(f"Zero-strength mismatch for feature #{feature_idx}")
                print(f"    [{prompt[:30]}] mean_act={result['mean_post_activation']:.3f} | MATCHES baseline: {result['completion'][:80]}")
            else:
                result = generate_with_steering(model, sae, feature_idx, direction, strength, prompt)
                print(f"    [{prompt[:30]}] mean_act={result['mean_post_activation']:.3f} max_act={result['max_post_activation']:.3f} | {result['completion']}")

if __name__ == "__main__":
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    candidates_to_check = [3574, 754, 1492, 727, 1046]

    print("Computing activation statistics for all candidates in one pass...")
    candidate_stats = get_activation_statistics_multi(model, sae, candidates_to_check)

    for feature_idx in candidates_to_check:
        test_candidate(model, sae, feature_idx, stats=candidate_stats[feature_idx], strength_multiples=(0, 3, 5, 10))