import torch
import numpy as np
import os
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae
from model.find_top_features import find_top_activating_examples_with_model

def load_beavertails_split(seed: int = 42, discovery_n: int = 1000, validation_n: int = 500, test_n: int = 500):
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train").shuffle(seed=seed)

    harmful_all, safe_all = [], []
    seen_texts = set()

    for example_id, example in enumerate(dataset):
        text = example["response"].strip()
        if not text:
            continue

        normalized = " ".join(text.lower().split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)

        record = {"id": example_id, "text": text}
        (safe_all if example["is_safe"] else harmful_all).append(record)

    total_needed = discovery_n + validation_n + test_n
    if len(harmful_all) < total_needed or len(safe_all) < total_needed:
        raise ValueError(f"Not enough examples after dedup: need {total_needed} per class, have {len(harmful_all)} harmful / {len(safe_all)} safe")

    splits = {
        "discovery": (harmful_all[:discovery_n], safe_all[:discovery_n]),
        "validation": (harmful_all[discovery_n:discovery_n+validation_n], safe_all[discovery_n:discovery_n+validation_n]),
        "test": (harmful_all[discovery_n+validation_n:discovery_n+validation_n+test_n], safe_all[discovery_n+validation_n:discovery_n+validation_n+test_n]),
    }
    for name, (h, s) in splits.items():
        print(f"{name}: {len(h)} harmful, {len(s)} safe")
    return splits

def get_per_example_activations(model, sae, records, layer: int = 6, batch_size: int = 8):
    texts = [r["text"] for r in records]
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = model.to_tokens(batch_texts, prepend_bos=True, padding_side="right")

        lengths = torch.tensor([model.to_tokens(t, prepend_bos=True).shape[1] for t in batch_texts])
        positions = torch.arange(tokens.shape[1])
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post")
            acts = cache[f"blocks.{layer}.hook_resid_post"]
            features, _ = sae(acts.reshape(-1, acts.shape[-1]))
            features = features.reshape(acts.shape[0], acts.shape[1], -1)

        for b in range(len(batch_texts)):
            real_features = features[b][mask[b]]
            all_features.append(real_features.mean(dim=0))

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{len(texts)} texts")

    return torch.stack(all_features)

def compute_cohens_d_safe(harmful_features, safe_features, min_prevalence: float = 0.05):
    mean_h, mean_s = harmful_features.mean(dim=0), safe_features.mean(dim=0)
    var_h, var_s = harmful_features.var(dim=0, unbiased=True), safe_features.var(dim=0, unbiased=True)
    pooled_std = torch.sqrt((var_h + var_s) / 2)

    prevalence_h = (harmful_features > 0).float().mean(dim=0)
    prevalence_s = (safe_features > 0).float().mean(dim=0)
    fires_enough = (prevalence_h > min_prevalence) | (prevalence_s > min_prevalence)

    valid = (pooled_std > 1e-6) & fires_enough

    cohens_d = torch.full_like(pooled_std, float("-inf"))
    cohens_d[valid] = (mean_h[valid] - mean_s[valid]) / pooled_std[valid]

    return cohens_d, mean_h, mean_s, var_h.sqrt(), var_s.sqrt(), prevalence_h, prevalence_s

def evaluate_feature_as_monitor(feature_idx, val_harmful, val_safe, test_harmful, test_safe):
    val_scores = torch.cat([val_harmful[:, feature_idx], val_safe[:, feature_idx]]).numpy()
    val_labels = torch.cat([torch.ones(val_harmful.shape[0]), torch.zeros(val_safe.shape[0])]).numpy()

    val_precision, val_recall, thresholds = precision_recall_curve(val_labels, val_scores)
    if thresholds.size == 0:
        raise ValueError("Validation feature scores are constant — cannot select a threshold.")

    threshold_precision = val_precision[:-1]
    threshold_recall = val_recall[:-1]
    f1_values = (2 * threshold_precision * threshold_recall) / np.maximum(threshold_precision + threshold_recall, 1e-12)

    best_idx = int(np.argmax(f1_values))
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_values[best_idx])
    print(f"  Threshold selected on validation: {best_thresh:.4f} (val F1={best_f1:.4f})")

    test_scores = torch.cat([test_harmful[:, feature_idx], test_safe[:, feature_idx]]).numpy()
    test_labels = torch.cat([torch.ones(test_harmful.shape[0]), torch.zeros(test_safe.shape[0])]).numpy()
    test_preds = (test_scores >= best_thresh).astype(int)

    auroc = roc_auc_score(test_labels, test_scores)
    auprc = average_precision_score(test_labels, test_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary", zero_division=0)
    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n  --- Feature #{feature_idx} — FROZEN test-set results ---")
    print(f"  AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")

    return {
        "threshold": float(best_thresh),
        "validation_f1": float(best_f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

def inspect_candidate_with_threshold(model, sae, feature_idx, val_harmful_records, val_safe_records, val_harmful_acts, val_safe_acts, threshold, num_openweb_texts: int = 200, top_k: int = 5, layer: int = 6):
    print(f"\n=== Feature #{feature_idx} — general text (OpenWebText) ===")
    examples = find_top_activating_examples_with_model(model, sae, feature_idx=feature_idx, num_texts=num_openweb_texts, top_k=top_k, layer=layer)
    for val, context in examples:
        print(f"  {val:.2f} | {context}")

    h_scores = val_harmful_acts[:, feature_idx]
    s_scores = val_safe_acts[:, feature_idx]

    tp_idx = (h_scores >= threshold).nonzero().squeeze(-1)
    fn_idx = (h_scores < threshold).nonzero().squeeze(-1)
    fp_idx = (s_scores >= threshold).nonzero().squeeze(-1)

    print(f"\n=== True positives (harmful, score >= threshold) — validation set ===")
    for i in tp_idx[:5]:
        print(f"  {h_scores[i]:.2f} | {val_harmful_records[i]['text'][:150]}")

    print(f"\n=== False negatives (harmful, score < threshold) — validation set ===")
    for i in fn_idx[:5]:
        print(f"  {h_scores[i]:.2f} | {val_harmful_records[i]['text'][:150]}")

    print(f"\n=== False positives (safe, score >= threshold) — validation set ===")
    for i in fp_idx[:5]:
        print(f"  {s_scores[i]:.2f} | {val_safe_records[i]['text'][:150]}")

def find_harm_features(sae_weights_path: str = "model/sae_weights.pt", layer: int = 6, top_n: int = 10, seed: int = 42):
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    model = load_base_model(device="cpu")
    sae = load_trained_sae(sae_weights_path).to("cpu")
    sae.eval()

    if top_n > sae.W_enc.shape[0]:
        raise ValueError("top_n must not exceed the number of SAE features.")

    splits = load_beavertails_split(seed=seed)
    os.makedirs("results", exist_ok=True)

    config = {
        "dataset": "PKU-Alignment/BeaverTails", "source_split": "30k_train", "seed": seed,
        "layer": layer, "aggregation": "mean_token_activation", "minimum_prevalence": 0.05,
        "discovery_per_class": 1000, "validation_per_class": 500, "test_per_class": 500,
    }

    disc_h, disc_s = splits["discovery"]
    print("\nComputing DISCOVERY activations...")
    disc_harmful = get_per_example_activations(model, sae, disc_h, layer)
    disc_safe = get_per_example_activations(model, sae, disc_s, layer)

    cohens_d, mean_h, mean_s, std_h, std_s, prev_h, prev_s = compute_cohens_d_safe(disc_harmful, disc_safe)

    num_valid = torch.isfinite(cohens_d).sum().item()
    if num_valid < top_n:
        raise ValueError(f"Only {num_valid} features passed selection filters, but top_n={top_n}.")

    top_values, top_indices = torch.topk(cohens_d, k=top_n)

    print(f"\nTop {top_n} candidates (discovery only, Cohen's d, min-prevalence filtered):")
    for val, idx in zip(top_values, top_indices):
        i = idx.item()
        print(f"  Feature #{i}: d={val.item():.4f} | harm_prev={prev_h[i]:.3f} | safe_prev={prev_s[i]:.3f}")

    torch.save({
        "config": config, "cohens_d": cohens_d, "mean_harmful": mean_h, "mean_safe": mean_s,
        "std_harmful": std_h, "std_safe": std_s, "prevalence_harmful": prev_h, "prevalence_safe": prev_s,
        "top_indices": top_indices, "top_values": top_values,
        "discovery_harmful_ids": [r["id"] for r in disc_h], "discovery_safe_ids": [r["id"] for r in disc_s],
        "discovery_harmful_activations": disc_harmful, "discovery_safe_activations": disc_safe,
    }, "results/harm_feature_discovery.pt")

    del disc_harmful, disc_safe

    frozen_feature = top_indices[0].item()
    print(f"\n>>> FROZEN candidate for validation/test: Feature #{frozen_feature} <<<")

    val_h, val_s = splits["validation"]
    print("\nComputing VALIDATION activations...")
    val_harmful = get_per_example_activations(model, sae, val_h, layer)
    val_safe = get_per_example_activations(model, sae, val_s, layer)

    test_h, test_s = splits["test"]
    print("\nComputing TEST activations...")
    test_harmful = get_per_example_activations(model, sae, test_h, layer)
    test_safe = get_per_example_activations(model, sae, test_s, layer)

    monitor_results = evaluate_feature_as_monitor(frozen_feature, val_harmful, val_safe, test_harmful, test_safe)

    torch.save({
        "config": config, "frozen_feature": frozen_feature, "monitor_results": monitor_results,
        "validation_harmful_ids": [r["id"] for r in val_h], "validation_safe_ids": [r["id"] for r in val_s],
        "test_harmful_ids": [r["id"] for r in test_h], "test_safe_ids": [r["id"] for r in test_s],
        "validation_harmful_activations": val_harmful, "validation_safe_activations": val_safe,
        "test_harmful_activations": test_harmful, "test_safe_activations": test_safe,
    }, "results/harm_feature_monitor_eval.pt")

    print("\n\nQUALITATIVE INSPECTION (threshold-aware, on VALIDATION set):")
    inspect_candidate_with_threshold(model, sae, frozen_feature, val_h, val_s, val_harmful, val_safe, monitor_results["threshold"])

    return frozen_feature, monitor_results

if __name__ == "__main__":
    find_harm_features()