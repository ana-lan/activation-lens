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

def get_matched_random_direction(dim=768, seed=42):
    """A random unit direction with the same norm convention as our SAE
    decoder directions — the essential control for distinguishing
    direction-specific effects from generic residual perturbation."""
    gen = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=gen)
    return direction / direction.norm()

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

def generate_with_persistent_steering(model, direction, strength, prompt, max_new_tokens=25, layer=6):
    """Applies the steering direction to EVERY position in the residual
    stream, not just the last one — closer to Golden Gate Claude's
    persistent, all-position intervention."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    prompt_len = tokens.shape[1]

    def steering_hook(residual, hook):
        modified = residual.clone()
        modified += strength * direction.view(1, 1, -1)
        return modified

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, steering_hook)])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == model.tokenizer.eos_token_id:
            break

    completion_tokens = tokens[0, prompt_len:]
    return {"completion": model.to_string(completion_tokens), "completion_token_ids": completion_tokens.cpu().tolist()}

def test_persistent(model, sae, feature_idx, stats, direction=None, label=None, strength_multiples=(0, 3, 5, 10)):
    if direction is None:
        direction, dec_norm = get_steering_direction(sae, feature_idx)
    else:
        dec_norm = 1.0  # random direction already unit-normalized

    activation_reference = stats["p99_all"] if stats["p99_all"] > 1e-8 else stats["p95_active"]

    print(f"\n{'='*70}")
    print(f"{label} | decoder_norm={dec_norm:.4f}")
    print(f"{'='*70}")

    for mult in strength_multiples:
        strength = mult * activation_reference * dec_norm
        print(f"\n  --- strength = {mult}x reference ({strength:.4f}) ---")
        for prompt in NEUTRAL_PROMPTS:
            if mult == 0:
                baseline = generate_baseline(model, prompt)
                result = generate_with_persistent_steering(model, direction, 0.0, prompt)
                if result["completion_token_ids"] != baseline["completion_token_ids"]:
                    raise AssertionError(f"Zero-strength mismatch for {label}")
                print(f"    [{prompt[:30]}] MATCHES baseline: {result['completion'][:80]}")
            else:
                result = generate_with_persistent_steering(model, direction, strength, prompt)
                print(f"    [{prompt[:30]}] {result['completion']}")

if __name__ == "__main__":
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    candidates_to_check = [3574, 754, 727]

    print("Computing activation statistics for all candidates in one pass...")
    candidate_stats = get_activation_statistics_multi(model, sae, candidates_to_check)

    for feature_idx in candidates_to_check:
        direction, _ = get_steering_direction(sae, feature_idx)
        label = f"Feature #{feature_idx} (PERSISTENT, all-position)"
        test_persistent(model, sae, feature_idx, candidate_stats[feature_idx], direction=direction, label=label)

    # matched random-direction control, using #3574's activation scale as reference
    random_direction = get_matched_random_direction(seed=42)
    test_persistent(model, sae, None, candidate_stats[3574], direction=random_direction, label="RANDOM unit direction (matched to #3574 scale)")