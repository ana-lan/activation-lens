import torch
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae
from safety.find_harm_feature import load_beavertails_split

_clf_data = torch.load("results/multi_feature_classifier.pt")
SELECTED_FEATURES = _clf_data["selected_features"]
COEFFICIENTS = np.array(_clf_data["coefficients"])
INTERCEPT = _clf_data["intercept"]
SCALER_MEAN = np.array(_clf_data["scaler_mean"])
SCALER_SCALE = np.array(_clf_data["scaler_scale"])
MIN_PREFIX_TOKENS = 5
SHORT_RESPONSE_POLICY = "score_at_end"  # if a response ends before 5 tokens, score its final available prefix once

def score_running_mean(running_mean_selected_features):
    scaled = (running_mean_selected_features - SCALER_MEAN) / SCALER_SCALE
    logit = np.dot(scaled, COEFFICIENTS) + INTERCEPT
    return float(expit(logit))

def get_max_prefix_score(model, sae, text, layer: int = 6):
    """One GPT-2 forward pass gives activations at every position (causal
    model), so we get every prefix's activation cheaply from a single pass —
    not one pass per prefix. BOS (position 0) is excluded so this matches
    the live monitor's running mean, which only ever sees generated tokens."""
    tokens = model.to_tokens(text, prepend_bos=True)
    hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name][0, 1:]  # exclude BOS at position 0
        features, _ = sae(acts)

    selected = features[:, SELECTED_FEATURES].cpu().numpy()  # [response_len, 100]

    running_sum = np.zeros(len(SELECTED_FEATURES))
    max_score = -np.inf
    reached_min_length = False

    for t in range(selected.shape[0]):
        running_sum += selected[t]
        num_seen = t + 1
        if num_seen < MIN_PREFIX_TOKENS:
            continue
        reached_min_length = True
        running_mean = running_sum / num_seen
        max_score = max(max_score, score_running_mean(running_mean))

    if not reached_min_length:
        # SHORT_RESPONSE_POLICY = "score_at_end": response never reached
        # MIN_PREFIX_TOKENS, so score its final available prefix once.
        num_seen = max(selected.shape[0], 1)
        running_mean = running_sum / num_seen
        max_score = score_running_mean(running_mean)

    return max_score

def compute_max_prefix_scores(model, sae, records, layer: int = 6):
    scores = []
    for i, r in enumerate(records):
        scores.append(get_max_prefix_score(model, sae, r["text"], layer))
        if i % 50 == 0:
            print(f"  Processed {i}/{len(records)} texts")
    return np.array(scores)

def calibrate_and_evaluate_live_rule():
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    splits = load_beavertails_split(seed=42)
    val_h, val_s = splits["validation"]

    print("Computing max-prefix scores on VALIDATION (calibration only)...")
    val_h_scores = compute_max_prefix_scores(model, sae, val_h)
    val_s_scores = compute_max_prefix_scores(model, sae, val_s)

    val_scores = np.concatenate([val_h_scores, val_s_scores])
    val_labels = np.concatenate([np.ones(len(val_h_scores)), np.zeros(len(val_s_scores))])

    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_scores)
    eligible = np.where(recalls[:-1] >= 0.90)[0]
    if eligible.size == 0:
        raise ValueError("No validation threshold achieves at least 90% recall.")
    selected_idx = eligible[np.argmax(precisions[:-1][eligible])]

    live_threshold = float(thresholds[selected_idx])
    validation_precision = float(precisions[:-1][selected_idx])
    validation_recall = float(recalls[:-1][selected_idx])
    print(f"\nCalibrated live threshold (validation, min_prefix={MIN_PREFIX_TOKENS}): {live_threshold:.4f}")
    print(f"Validation precision={validation_precision:.4f} | recall={validation_recall:.4f}")

    from safety.multi_feature_classifier import get_final_holdout
    holdout_h, holdout_s = get_final_holdout(offset=3000, holdout_n=500)

    print("\nComputing max-prefix scores on FINAL HOLDOUT (frozen evaluation)...")
    test_h_scores = compute_max_prefix_scores(model, sae, holdout_h)
    test_s_scores = compute_max_prefix_scores(model, sae, holdout_s)

    test_scores = np.concatenate([test_h_scores, test_s_scores])
    test_labels = np.concatenate([np.ones(len(test_h_scores)), np.zeros(len(test_s_scores))])
    test_preds = (test_scores >= live_threshold).astype(int)

    auroc = roc_auc_score(test_labels, test_scores)
    auprc = average_precision_score(test_labels, test_scores)
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp = cm[0]
    fn, tp = cm[1]
    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"\n--- FROZEN live-monitor results (offset 3000:3500) ---")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | FPR: {fpr:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

    torch.save({
        "min_prefix_tokens": MIN_PREFIX_TOKENS,
        "short_response_policy": SHORT_RESPONSE_POLICY,
        "live_threshold": live_threshold,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "live_auroc": float(auroc),
        "live_auprc": float(auprc),
        "live_precision": float(precision),
        "live_recall": float(recall),
        "live_fpr": float(fpr),
        "confusion_matrix": cm.tolist(),
        "holdout_offset": 3000,
        "distribution_limitation": "Calibrated on standalone BeaverTails responses; live-generated tokens are conditioned on a prompt, a real distribution shift not fully validated here.",
    }, "results/live_multifeature_calibration.pt")

    return live_threshold

if __name__ == "__main__":
    calibrate_and_evaluate_live_rule()