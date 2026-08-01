import torch
import numpy as np
import os
from datasets import load_dataset
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

def build_streaming_topk_all_features(model, sae, layer=6, start=10200, num_texts=150, top_k=4):
    """Streams through documents once, keeping only small [top_k, 8192] arrays
    (values, doc_ids, positions) — never retains full per-document activation
    matrices, avoiding a large memory blowup from caching everything."""
    dataset = load_dataset("Skylion007/openwebtext", split=f"train[{start}:{start+num_texts}]")
    hook_name = f"blocks.{layer}.hook_resid_post"

    num_features = sae.W_enc.shape[0]
    top_values = torch.full((top_k, num_features), float("-inf"))
    top_doc_ids = torch.full((top_k, num_features), -1, dtype=torch.long)
    top_positions = torch.full((top_k, num_features), -1, dtype=torch.long)

    all_str_tokens = []
    prevalence_counts = torch.zeros(num_features)
    total_tokens = 0

    for doc_idx, example in enumerate(dataset):
        text = example["text"]
        if not text.strip():
            all_str_tokens.append([])
            continue

        tokens = model.to_tokens(text, prepend_bos=True)
        str_tokens = model.to_str_tokens(text, prepend_bos=True)[1:]  # exclude BOS
        all_str_tokens.append(str_tokens)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n == hook_name,
                return_type=None,  # skip computing/retaining logits entirely
            )
            acts = cache[hook_name][0, 1:]
            features, _ = sae(acts)

        prevalence_counts += (features > 0).sum(dim=0)
        total_tokens += features.shape[0]

        doc_max_vals, doc_max_pos = features.max(dim=0)

        combined_vals = torch.cat([top_values, doc_max_vals.unsqueeze(0)], dim=0)
        combined_doc_ids = torch.cat([top_doc_ids, torch.full((1, num_features), doc_idx, dtype=torch.long)], dim=0)
        combined_positions = torch.cat([top_positions, doc_max_pos.unsqueeze(0)], dim=0)

        new_top_vals, sel = torch.topk(combined_vals, k=top_k, dim=0)
        new_top_doc_ids = torch.gather(combined_doc_ids, 0, sel)
        new_top_positions = torch.gather(combined_positions, 0, sel)

        top_values, top_doc_ids, top_positions = new_top_vals, new_top_doc_ids, new_top_positions

        del features, acts, cache

        if doc_idx % 30 == 0:
            print(f"  Processed {doc_idx}/{num_texts} texts")

    prevalence = prevalence_counts / max(total_tokens, 1)
    return top_values, top_doc_ids, top_positions, all_str_tokens, prevalence

def get_top_examples_from_streamed(top_values, top_doc_ids, top_positions, all_str_tokens, feature_idx, top_k=4):
    results = []
    for k in range(top_k):
        val = top_values[k, feature_idx].item()
        doc_idx = top_doc_ids[k, feature_idx].item()
        pos = top_positions[k, feature_idx].item()
        if doc_idx < 0:
            continue
        str_tokens = all_str_tokens[doc_idx]
        start_ctx = max(0, pos - 5)
        end_ctx = min(len(str_tokens), pos + 6)
        context = "".join(str_tokens[start_ctx:end_ctx])
        results.append((val, context))
    results.sort(key=lambda x: x[0], reverse=True)
    return results

def build_diverse_candidate_pool(prevalence, seed=42, n_random=40, n_moderate_prevalence=20, phase4_features=None):
    """Mixes several selection strategies rather than pure random sampling.
    Note: moderate-prevalence features are still randomly sampled within
    that band — a prevalence-stratified sample, not a guarantee of
    semantic cleanliness."""
    rng = np.random.default_rng(seed)
    num_features = prevalence.shape[0]

    random_pool = rng.choice(num_features, size=n_random, replace=False).tolist()

    moderate_mask = (prevalence > 0.001) & (prevalence < 0.5)
    moderate_candidates = torch.where(moderate_mask)[0].tolist()
    moderate_pool = rng.choice(moderate_candidates, size=min(n_moderate_prevalence, len(moderate_candidates)), replace=False).tolist()

    phase4_pool = phase4_features or [0, 10, 100, 500, 1000, 4000]

    combined = sorted(set(random_pool + moderate_pool + phase4_pool))
    print(f"Candidate pool: {len(random_pool)} random + {len(moderate_pool)} moderate-prevalence + {len(phase4_pool)} Phase 4 features "
          f"= {len(combined)} unique candidates")
    return combined

if __name__ == "__main__":
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    print("Building streaming top-k cache across all 8192 features (one pass)...")
    top_values, top_doc_ids, top_positions, all_str_tokens, prevalence = build_streaming_topk_all_features(
        model, sae, num_texts=150, top_k=4
    )

    os.makedirs("results", exist_ok=True)
    torch.save({
        "top_values": top_values, "top_doc_ids": top_doc_ids, "top_positions": top_positions,
        "all_str_tokens": all_str_tokens, "prevalence": prevalence,
        "dataset_start": 10200, "num_texts": 150, "top_k": 4, "layer": 6,
    }, "results/steering_candidate_screen.pt")

    candidate_pool = build_diverse_candidate_pool(prevalence)

    print(f"\nScreening {len(candidate_pool)} candidates for concrete, nameable concepts...")
    for feature_idx in candidate_pool:
        print(f"\n--- Feature #{feature_idx} (prevalence={prevalence[feature_idx]:.4f}) ---")
        examples = get_top_examples_from_streamed(top_values, top_doc_ids, top_positions, all_str_tokens, feature_idx)
        for val, context in examples:
            print(f"  {val:.2f} | {context}")