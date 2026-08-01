import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_recall_fscore_support, confusion_matrix
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

def get_per_token_max_activation(model, sae, records, feature_idx, layer: int = 6, batch_size: int = 8):
    """Unlike Phase 6.2's mean-per-document, this tracks the MAX single-token
    activation per document — matching what the live monitor actually checks."""
    texts = [r["text"] for r in records]
    max_scores = []

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
            real_scores = features[b][mask[b]][:, feature_idx]
            max_scores.append(real_scores.max().item())

    return torch.tensor(max_scores)

def calibrate_live_threshold(sae_weights_path: str = "model/sae_weights.pt", layer: int = 6):
    model = load_base_model(device="cpu")
    sae = load_trained_sae(sae_weights_path).to("cpu")
    sae.eval()

    from safety.find_harm_feature import load_beavertails_split
    splits = load_beavertails_split(seed=42)
    frozen_feature = torch.load("results/harm_feature_monitor_eval.pt")["frozen_feature"]

    val_h, val_s = splits["validation"]
    test_h, test_s = splits["test"]

    print("Computing MAX-token activations (validation)...")
    val_h_scores = get_per_token_max_activation(model, sae, val_h, frozen_feature, layer)
    val_s_scores = get_per_token_max_activation(model, sae, val_s, frozen_feature, layer)

    print("Computing MAX-token activations (test)...")
    test_h_scores = get_per_token_max_activation(model, sae, test_h, frozen_feature, layer)
    test_s_scores = get_per_token_max_activation(model, sae, test_s, frozen_feature, layer)

    val_scores = torch.cat([val_h_scores, val_s_scores]).numpy()
    val_labels = torch.cat([torch.ones(len(val_h_scores)), torch.zeros(len(val_s_scores))]).numpy()

    precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
    f1s = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.argmax(f1s))
    live_threshold = float(thresholds[best_idx])

    test_scores = torch.cat([test_h_scores, test_s_scores]).numpy()
    test_labels = torch.cat([torch.ones(len(test_h_scores)), torch.zeros(len(test_s_scores))]).numpy()
    test_preds = (test_scores >= live_threshold).astype(int)

    auroc = roc_auc_score(test_labels, test_scores)
    auprc = average_precision_score(test_labels, test_scores)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary", zero_division=0)
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp = cm[0]
    fpr = fp / (fp + tn)

    print(f"\nLIVE (max-token) monitor — frozen threshold: {live_threshold:.4f}")
    print(f"Test AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | FPR: {fpr:.4f}")

    torch.save({
        "live_threshold": float(live_threshold),
        "live_test_auroc": float(auroc),
        "live_test_auprc": float(auprc),
        "live_test_precision": float(p),
        "live_test_recall": float(r),
        "live_test_fpr": float(fpr),
        "confusion_matrix": cm.tolist(),
    }, "results/harm_feature_live_calibration.pt")

    return live_threshold

if __name__ == "__main__":
    calibrate_live_threshold()