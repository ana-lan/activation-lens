import torch
from datasets import load_dataset
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae
from model.quantize_model import quantize_model

def get_test_texts(num_texts: int = 500):
    dataset = load_dataset("Skylion007/openwebtext", split=f"train[:{num_texts}]")
    return [ex["text"] for ex in dataset if ex["text"].strip()]

def get_streaming_topk(model, sae, texts, layer: int = 6, batch_size: int = 8, top_k: int = 5):
    """Tracks only each feature's top-k activating positions — never holds the full dataset in memory."""
    running_values = None   # [k, num_features]
    running_global_idx = None  # [k, num_features] — global token position, not per-batch
    running_max = None      # [num_features] — for the "is this feature alive at all" check

    token_offset = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = model.to_tokens(batch_texts, prepend_bos=True, padding_side="right")

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == f"blocks.{layer}.hook_resid_post"
            )
            acts = cache[f"blocks.{layer}.hook_resid_post"]
            acts = acts.reshape(-1, acts.shape[-1])  # [batch_tokens, 8192]
            features, _ = sae(acts)  # [batch_tokens, 8192]

        num_tokens_this_batch = features.shape[0]

        # local top-k for this batch only
        k_this = min(top_k, num_tokens_this_batch)
        local_values, local_idx = torch.topk(features, k=k_this, dim=0)
        local_global_idx = local_idx + token_offset

        batch_max = features.max(dim=0).values

        if running_values is None:
            running_values = local_values
            running_global_idx = local_global_idx
            running_max = batch_max
        else:
            # merge running top-k with this batch's top-k, then re-select the overall top-k
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

    return running_global_idx, running_max  # [k, num_features], [num_features]

def compute_survival_rate(baseline_idx, baseline_max, variant_idx):
    """Vectorized — no Python loop over features."""
    matches = (baseline_idx.unsqueeze(1) == variant_idx.unsqueeze(0)).any(dim=0).any(dim=0)  # [num_features]
    alive_mask = baseline_max > 0

    survived = (matches & alive_mask).sum().item()
    total_alive = alive_mask.sum().item()
    return survived / total_alive if total_alive > 0 else 0.0

def compare_variants(sae_weights_path: str = "model/sae_weights.pt", layer: int = 6, num_texts: int = 500, top_k: int = 5):
    sae = load_trained_sae(sae_weights_path)
    texts = get_test_texts(num_texts)

    print("Computing baseline (full precision) top-k features...")
    baseline_model = load_base_model().to("cpu")
    baseline_idx, baseline_max = get_streaming_topk(baseline_model, sae, texts, layer, top_k=top_k)

    variant_bits = {"16bit": 16, "8bit": 8, "4bit": 4}

    for variant_name, bits in variant_bits.items():
        print(f"\nComputing {variant_name}...")
        model = quantize_model(load_base_model(), num_bits=bits).to("cpu")
        variant_idx, _ = get_streaming_topk(model, sae, texts, layer, top_k=top_k)

        survival_rate = compute_survival_rate(baseline_idx, baseline_max, variant_idx)
        print(f"=== {variant_name} === Feature survival rate: {survival_rate:.4f}")

if __name__ == "__main__":
    compare_variants(num_texts=500)