import torch
import torch.nn.functional as F
import math
import os
from datasets import load_dataset
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae
from model.quantize_model import quantize_model

def get_test_texts(start: int = 10000, num_texts: int = 500):
    end = start + num_texts
    dataset = load_dataset("Skylion007/openwebtext", split=f"train[{start}:{end}]")
    texts = [ex["text"] for ex in dataset if ex["text"].strip()]
    print(f"Evaluating on {len(texts)} non-empty held-out documents")
    return texts

def tokenize_with_mask(model, texts):
    tokens = model.to_tokens(texts, prepend_bos=True, padding_side="right")
    lengths = torch.tensor(
        [model.to_tokens(text, prepend_bos=True).shape[1] for text in texts],
        device=tokens.device,
    )
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    return tokens, mask

def get_streaming_topk(model, sae, texts, layer: int = 6, batch_size: int = 8, top_k: int = 5):
    running_values = None
    running_global_idx = None
    running_max = None
    token_offset = 0
    hook_name = f"blocks.{layer}.hook_resid_post"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens, mask = tokenize_with_mask(model, batch_texts)
        mask_flat = mask.reshape(-1)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            acts = cache[hook_name].reshape(-1, cache[hook_name].shape[-1])
            features = sae(acts)[0]

        features = features[mask_flat]
        num_tokens_this_batch = features.shape[0]
        if num_tokens_this_batch == 0:
            continue

        k_this = min(top_k, num_tokens_this_batch)
        local_values, local_idx = torch.topk(features, k=k_this, dim=0)
        local_global_idx = local_idx + token_offset
        batch_max = features.max(dim=0).values

        if running_values is None:
            running_values = local_values
            running_global_idx = local_global_idx
            running_max = batch_max
        else:
            combined_values = torch.cat([running_values, local_values], dim=0)
            combined_idx = torch.cat([running_global_idx, local_global_idx], dim=0)
            k_final = min(top_k, combined_values.shape[0])
            new_values, sel = torch.topk(combined_values, k=k_final, dim=0)
            new_idx = torch.gather(combined_idx, 0, sel)
            running_values = new_values
            running_global_idx = new_idx
            running_max = torch.maximum(running_max, batch_max)

        token_offset += num_tokens_this_batch
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{len(texts)} texts")

    return running_global_idx, running_values, running_max

def compute_topk_metrics(baseline_idx, baseline_values, variant_idx, variant_values, activation_eps: float = 1e-8):
    matches = (baseline_idx.unsqueeze(1) == variant_idx.unsqueeze(0))
    baseline_positive = baseline_values > activation_eps
    variant_positive = variant_values > activation_eps

    valid_matches = matches & baseline_positive.unsqueeze(1) & variant_positive.unsqueeze(0)
    matched_baseline_positions = valid_matches.any(dim=1)

    positive_baseline_count = baseline_positive.sum(dim=0)
    matched_count = matched_baseline_positions.sum(dim=0)
    active_mask = positive_baseline_count > 0

    per_feature_recall = torch.zeros_like(positive_baseline_count, dtype=torch.float32)
    per_feature_recall[active_mask] = matched_count[active_mask].float() / positive_baseline_count[active_mask].float()

    return {
        "mean_recall_at_k": per_feature_recall[active_mask].mean().item(),
        "any_hit_rate": (matched_count[active_mask] > 0).float().mean().item(),
        "num_active_features": active_mask.sum().item(),
    }

def compute_correlation_streaming(baseline_model, variant_model, sae, texts, layer: int = 6, batch_size: int = 8):
    num_features = sae.W_enc.shape[0]
    sum_b = torch.zeros(num_features, dtype=torch.float64)
    sum_v = torch.zeros(num_features, dtype=torch.float64)
    sum_b2 = torch.zeros(num_features, dtype=torch.float64)
    sum_v2 = torch.zeros(num_features, dtype=torch.float64)
    sum_bv = torch.zeros(num_features, dtype=torch.float64)
    baseline_max = torch.full((num_features,), float("-inf"), dtype=torch.float64)
    count = 0
    hook_name = f"blocks.{layer}.hook_resid_post"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens, mask = tokenize_with_mask(baseline_model, batch_texts)
        mask_flat = mask.reshape(-1)

        with torch.no_grad():
            _, cache_b = baseline_model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            acts_b = cache_b[hook_name].reshape(-1, cache_b[hook_name].shape[-1])
            feat_b = sae(acts_b)[0][mask_flat].double()

            _, cache_v = variant_model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            acts_v = cache_v[hook_name].reshape(-1, cache_v[hook_name].shape[-1])
            feat_v = sae(acts_v)[0][mask_flat].double()

        count += feat_b.shape[0]
        sum_b += feat_b.sum(dim=0)
        sum_v += feat_v.sum(dim=0)
        sum_b2 += (feat_b * feat_b).sum(dim=0)
        sum_v2 += (feat_v * feat_v).sum(dim=0)
        sum_bv += (feat_b * feat_v).sum(dim=0)
        baseline_max = torch.maximum(baseline_max, feat_b.max(dim=0).values)

        if (i // batch_size) % 10 == 0:
            print(f"  Correlation: processed {i}/{len(texts)} texts")

    covariance = sum_bv - (sum_b * sum_v / count)
    variance_b = sum_b2 - (sum_b.square() / count)
    variance_v = sum_v2 - (sum_v.square() / count)
    denominator = torch.sqrt(variance_b.clamp_min(0) * variance_v.clamp_min(0))

    baseline_variable = variance_b > 1e-12
    variant_variable = variance_v > 1e-12
    baseline_active = baseline_max > 0

    eligible_mask = baseline_active & baseline_variable
    valid_mask = eligible_mask & variant_variable & (denominator > 1e-12)
    collapsed_mask = eligible_mask & ~variant_variable

    correlations = torch.full((num_features,), float("nan"), dtype=torch.float64)
    correlations[valid_mask] = covariance[valid_mask] / denominator[valid_mask]

    return {
        "correlations": correlations,
        "eligible_mask": eligible_mask,
        "valid_mask": valid_mask,
        "collapsed_mask": collapsed_mask,
    }

def report_correlation_stats(result, variant_name):
    correlations = result["correlations"]
    eligible = result["eligible_mask"]
    valid_mask = result["valid_mask"]
    collapsed = result["collapsed_mask"]
    valid = correlations[valid_mask]

    eligible_count = eligible.sum().item()
    valid_count = valid_mask.sum().item()
    collapsed_count = collapsed.sum().item()

    print(f"\n--- {variant_name} correlation distribution ---")
    print(f"  Eligible baseline features: {eligible_count}")
    print(f"  Features with valid correlation: {valid_count}")
    print(f"  Features collapsed after quantization: {collapsed_count / max(eligible_count, 1):.4f}")

    if valid.numel() == 0:
        print("  No features had valid correlation.")
        return

    print(f"  Mean correlation:   {valid.mean().item():.4f}")
    print(f"  Median correlation: {valid.median().item():.4f}")
    print(f"  10th percentile:    {torch.quantile(valid, 0.10).item():.4f}")
    print(f"  Features r >= .90:  {(valid >= 0.90).float().mean().item():.4f}")
    print(f"  Features r >= .80:  {(valid >= 0.80).float().mean().item():.4f}")
    print(f"  Features r < .50:   {(valid < 0.50).float().mean().item():.4f}")

def safe_exp(value):
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")

def compare_behavior(baseline_model, variant_model, texts, batch_size=1):
    total_baseline_loss = 0.0
    total_variant_loss = 0.0
    total_kl = 0.0
    total_agreement = 0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens, mask = tokenize_with_mask(baseline_model, batch_texts)

        prediction_mask = mask[:, 1:]
        targets = tokens[:, 1:]

        with torch.no_grad():
            baseline_logits = baseline_model(tokens)
        baseline_predictions = baseline_logits[:, :-1]
        baseline_loss = F.cross_entropy(
            baseline_predictions.reshape(-1, baseline_predictions.shape[-1]),
            targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        baseline_top1 = baseline_predictions.argmax(dim=-1)
        baseline_log_probs = F.log_softmax(baseline_predictions, dim=-1)
        del baseline_logits, baseline_predictions

        with torch.no_grad():
            variant_logits = variant_model(tokens)
        variant_predictions = variant_logits[:, :-1]
        variant_loss = F.cross_entropy(
            variant_predictions.reshape(-1, variant_predictions.shape[-1]),
            targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        variant_top1 = variant_predictions.argmax(dim=-1)
        variant_log_probs = F.log_softmax(variant_predictions, dim=-1)
        del variant_logits, variant_predictions

        baseline_probs = baseline_log_probs.exp()
        kl = (baseline_probs * (baseline_log_probs - variant_log_probs)).sum(dim=-1)
        agreement = baseline_top1 == variant_top1

        total_baseline_loss += baseline_loss[prediction_mask].sum().item()
        total_variant_loss += variant_loss[prediction_mask].sum().item()
        total_kl += kl[prediction_mask].sum().item()
        total_agreement += agreement[prediction_mask].sum().item()
        total_tokens += prediction_mask.sum().item()

        del baseline_log_probs, variant_log_probs, baseline_probs
        del baseline_loss, variant_loss, baseline_top1, variant_top1
        del kl, agreement

    if total_tokens == 0:
        raise ValueError("No prediction tokens available for behavioral evaluation.")

    baseline_ce = total_baseline_loss / total_tokens
    variant_ce = total_variant_loss / total_tokens

    return {
        "baseline_perplexity": safe_exp(baseline_ce),
        "variant_perplexity": safe_exp(variant_ce),
        "mean_kl": total_kl / total_tokens,
        "top1_agreement": total_agreement / total_tokens,
    }

def compare_variants(sae_weights_path: str = "model/sae_weights.pt", layer: int = 6, num_texts: int = 500, top_k: int = 5, behavior_num_texts: int = 40):
    sae = load_trained_sae(sae_weights_path).to("cpu")
    sae.eval()
    texts = get_test_texts(start=10000, num_texts=num_texts)

    if not texts:
        raise ValueError("No evaluation texts were loaded.")
    if behavior_num_texts <= 0:
        raise ValueError("behavior_num_texts must be positive.")

    os.makedirs("results", exist_ok=True)

    print("Computing baseline (full precision) top-k features...")
    baseline_model = load_base_model(device="cpu")
    baseline_idx, baseline_values, baseline_max = get_streaming_topk(baseline_model, sae, texts, layer, top_k=top_k)

    variant_bits = {"16bit": 16, "8bit": 8, "4bit": 4}

    for variant_name, bits in variant_bits.items():
        print(f"\nComputing {variant_name}...")
        model = load_base_model(device="cpu")
        model = quantize_model(model, num_bits=bits)

        variant_idx, variant_values, _ = get_streaming_topk(model, sae, texts, layer, top_k=top_k)
        topk_stats = compute_topk_metrics(baseline_idx, baseline_values, variant_idx, variant_values)

        print(f"=== {variant_name} ===")
        print(f"  Mean recall@{top_k}: {topk_stats['mean_recall_at_k']:.4f}")
        print(f"  Any-hit rate: {topk_stats['any_hit_rate']:.4f}")
        print(f"  Num active features evaluated: {topk_stats['num_active_features']}")

        corr_result = compute_correlation_streaming(baseline_model, model, sae, texts, layer)
        report_correlation_stats(corr_result, variant_name)

        print(f"  Computing behavioral comparison for {variant_name} (batch_size=1)...")
        behavior_texts = texts[:behavior_num_texts]
        behavior_stats = compare_behavior(baseline_model, model, behavior_texts, batch_size=1)

        perplexity_ratio = behavior_stats["variant_perplexity"] / behavior_stats["baseline_perplexity"]

        print(f"  --- {variant_name} behavioral comparison ---")
        print(f"  Baseline perplexity: {behavior_stats['baseline_perplexity']:.4f}")
        print(f"  {variant_name} perplexity: {behavior_stats['variant_perplexity']:.4f}")
        print(f"  Perplexity ratio: {perplexity_ratio:.4f}x")
        print(f"  Mean KL divergence: {behavior_stats['mean_kl']:.4f}")
        print(f"  Top-1 token agreement: {behavior_stats['top1_agreement']:.4f}")

        result = {
            "variant": variant_name, "bits": bits, "layer": layer, "top_k": top_k,
            "num_feature_texts": len(texts), "num_behavior_texts": len(behavior_texts),
            "topk_stats": topk_stats, "correlation": corr_result, "behavior": behavior_stats,
        }
        torch.save(result, f"results/{variant_name}_metrics.pt")

if __name__ == "__main__":
    compare_variants(num_texts=500)