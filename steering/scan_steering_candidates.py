import torch
import numpy as np
from datasets import load_dataset
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

def get_activation_statistics(model, sae, feature_idx, layer=6, start=10000, num_texts=50):
    end = start + num_texts
    dataset = load_dataset("Skylion007/openwebtext", split=f"train[{start}:{end}]")

    all_values = []
    hook_name = f"blocks.{layer}.hook_resid_post"

    for example in dataset:
        text = example["text"].strip()
        if not text:
            continue
        tokens = model.to_tokens(text, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            activations = cache[hook_name][0, 1:]
            features, _ = sae(activations)
        all_values.append(features[:, feature_idx].cpu())

    values = torch.cat(all_values)
    positive = values[values > 0]
    prevalence = positive.numel() / values.numel()

    return {
        "p99_all": torch.quantile(values, 0.99).item(),
        "p95_active": torch.quantile(positive, 0.95).item() if positive.numel() else 0.0,
        "prevalence": prevalence,
    }

def get_top_examples_held_out(model, sae, feature_idx, layer=6, start=10100, num_texts=100, top_k=5):
    dataset = load_dataset("Skylion007/openwebtext", split=f"train[{start}:{start+num_texts}]")
    hook_name = f"blocks.{layer}.hook_resid_post"
    results = []

    for example in dataset:
        text = example["text"]
        if not text.strip():
            continue

        tokens = model.to_tokens(text, prepend_bos=True)
        str_tokens = model.to_str_tokens(text, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            acts = cache[hook_name][0, 1:]  # exclude BOS
            features, _ = sae(acts)

        real_str_tokens = str_tokens[1:]  # exclude BOS to stay aligned with acts

        feature_values = features[:, feature_idx]
        max_val, max_pos = torch.max(feature_values, dim=0)

        start_ctx = max(0, max_pos.item() - 5)
        end_ctx = min(len(real_str_tokens), max_pos.item() + 6)
        context = "".join(real_str_tokens[start_ctx:end_ctx])
        results.append((max_val.item(), context))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def get_steering_direction(sae, feature_idx):
    direction = sae.W_dec[:, feature_idx].clone()
    assert torch.isfinite(direction).all(), f"Feature {feature_idx} decoder direction has non-finite values"
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
    return {
        "completion": model.to_string(completion_tokens),
        "completion_token_ids": completion_tokens.cpu().tolist(),
    }

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

def scan_candidate(model, sae, feature_idx, layer=6, strength_multiples=(0, 1, 3)):
    num_features = sae.W_enc.shape[0]
    if not 0 <= feature_idx < num_features:
        raise ValueError(f"Invalid feature index {feature_idx}")

    direction, dec_norm = get_steering_direction(sae, feature_idx)
    stats = get_activation_statistics(model, sae, feature_idx, layer, start=10000, num_texts=50)

    activation_reference = stats["p99_all"]
    if activation_reference <= 1e-8:
        activation_reference = stats["p95_active"]
    if activation_reference <= 1e-8:
        print(f"\nFeature #{feature_idx}: never activated on held-out calibration data, skipping.")
        return

    print(f"\n{'='*70}")
    print(f"Feature #{feature_idx} | decoder_norm={dec_norm:.4f} | p99_all={stats['p99_all']:.4f} | "
          f"p95_active={stats['p95_active']:.4f} | prevalence={stats['prevalence']:.4f}")
    print(f"{'='*70}")

    print("Top HELD-OUT examples (candidate-selection data, not final-benchmark holdout):")
    examples = get_top_examples_held_out(model, sae, feature_idx, layer, start=10100, num_texts=100, top_k=5)
    for val, context in examples:
        print(f"  {val:.2f} | {context}")

    NEUTRAL_PROMPTS = [
        "My favorite thing to do on the weekend is",
        "Yesterday I went to the store and",
        "The most important skill in life is",
    ]

    for mult in strength_multiples:
        strength = mult * activation_reference * dec_norm
        print(f"\n  --- strength = {mult}x reference ({strength:.4f}) ---")
        for prompt in NEUTRAL_PROMPTS:
            if mult == 0:
                baseline = generate_baseline(model, prompt)
                result = generate_with_steering(model, sae, feature_idx, direction, 0.0, prompt)

                if result["completion_token_ids"] != baseline["completion_token_ids"]:
                    raise AssertionError(f"Zero-strength steering changed generation for feature #{feature_idx}.")

                print(f"    [{prompt[:35]}] mean_act={result['mean_post_activation']:.3f} | MATCHES baseline")
            else:
                result = generate_with_steering(model, sae, feature_idx, direction, strength, prompt)
                print(f"    [{prompt[:35]}] mean_act={result['mean_post_activation']:.3f} max_act={result['max_post_activation']:.3f} | {result['completion']}")

if __name__ == "__main__":
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    candidates_to_check = [6432, 0]

    for feature_idx in candidates_to_check:
        scan_candidate(model, sae, feature_idx, strength_multiples=(0, 3, 5, 10, 20))