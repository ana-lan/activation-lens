import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae
from safety.find_harm_feature import load_beavertails_split, get_per_example_activations

def get_final_holdout(seed: int = 42, offset: int = 2500, holdout_n: int = 500):
    from datasets import load_dataset
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

    holdout_h = harmful_all[offset:offset+holdout_n]
    holdout_s = safe_all[offset:offset+holdout_n]

    if len(holdout_h) != holdout_n or len(holdout_s) != holdout_n:
        raise ValueError(f"Insufficient final holdout data: {len(holdout_h)} harmful / {len(holdout_s)} safe; expected {holdout_n} each.")

    print(f"Final holdout: {len(holdout_h)} harmful, {len(holdout_s)} safe (genuinely unseen, offset {offset})")
    return holdout_h, holdout_s

def select_top_features_by_abs_effect(cohens_d, n_features):
    finite_mask = torch.isfinite(cohens_d)
    valid_indices = torch.where(finite_mask)[0]
    valid_effects = cohens_d[finite_mask]

    rank_order = torch.argsort(valid_effects.abs(), descending=True)
    selected = valid_indices[rank_order[:n_features]]

    assert torch.isfinite(cohens_d[selected]).all(), "Selected features must all be finite"
    assert selected.unique().numel() == selected.numel(), "Selected features must be unique"
    print(f"  Selected {n_features} feature IDs (first 10): {selected[:10].tolist()}")
    return selected

def make_classifier():
    return LogisticRegression(solver="saga", l1_ratio=1.0, C=1.0, max_iter=5000, random_state=42)

def check_convergence(clf, label=""):
    if clf.n_iter_.max() >= clf.max_iter:
        print(f"  Warning: logistic regression ({label}) reached max_iter — may not have converged.")

def sanity_check_single_feature(discovery_h, discovery_s, val_h, val_s, feature_idx=3149, expected_auroc=0.6512, tolerance=0.02):
    X_train = torch.cat([discovery_h[:, [feature_idx]], discovery_s[:, [feature_idx]]]).numpy()
    y_train = np.concatenate([np.ones(discovery_h.shape[0]), np.zeros(discovery_s.shape[0])])
    X_val = torch.cat([val_h[:, [feature_idx]], val_s[:, [feature_idx]]]).numpy()
    y_val = np.concatenate([np.ones(val_h.shape[0]), np.zeros(val_s.shape[0])])

    raw_auroc = roc_auc_score(y_val, X_val[:, 0])
    print(f"Raw #{feature_idx} validation AUROC (expected ~{expected_auroc}): {raw_auroc:.4f}")

    scaler = StandardScaler().fit(X_train)
    clf = make_classifier()
    clf.fit(scaler.transform(X_train), y_train)
    check_convergence(clf, "sanity check")
    clf_auroc = roc_auc_score(y_val, clf.predict_proba(scaler.transform(X_val))[:, 1])
    print(f"Classifier #{feature_idx} validation AUROC (expected ~{expected_auroc}): {clf_auroc:.4f}")

    if abs(raw_auroc - expected_auroc) > tolerance or abs(clf_auroc - expected_auroc) > tolerance:
        raise ValueError(
            f"SANITY CHECK FAILED — pipeline has a bug. "
            f"Expected ~{expected_auroc}, got raw={raw_auroc:.4f}, classifier={clf_auroc:.4f}. "
            f"Do not trust multi-feature results until this is fixed."
        )
    print("Sanity check PASSED — pipeline is trustworthy.\n")

def select_threshold_at_min_recall(precisions, recalls, thresholds, min_recall=0.90):
    eligible = np.where(recalls[:-1] >= min_recall)[0]
    if eligible.size == 0:
        raise ValueError(f"No validation threshold achieves at least {min_recall*100:.0f}% recall.")

    best_precision_position = np.argmax(precisions[:-1][eligible])
    threshold_idx = eligible[best_precision_position]

    threshold = float(thresholds[threshold_idx])
    selected_recall = float(recalls[threshold_idx])
    selected_precision = float(precisions[threshold_idx])

    print(f"Selected validation threshold={threshold:.4f} | precision={selected_precision:.4f} | recall={selected_recall:.4f}")
    return threshold

def train_and_evaluate_classifier(discovery_h, discovery_s, val_h, val_s, holdout_h_acts, holdout_s_acts, cohens_d, num_features_list=[10, 25, 50, 100]):
    X_disc = torch.cat([discovery_h, discovery_s]).numpy()
    y_disc = np.concatenate([np.ones(len(discovery_h)), np.zeros(len(discovery_s))])

    X_val = torch.cat([val_h, val_s]).numpy()
    y_val = np.concatenate([np.ones(len(val_h)), np.zeros(len(val_s))])

    X_holdout = torch.cat([holdout_h_acts, holdout_s_acts]).numpy()
    y_holdout = np.concatenate([np.ones(len(holdout_h_acts)), np.zeros(len(holdout_s_acts))])

    results = {}
    best_val_auroc = -1
    best_config = None

    for n_feat in num_features_list:
        feat_idx = select_top_features_by_abs_effect(cohens_d, n_feat).numpy()

        scaler = StandardScaler().fit(X_disc[:, feat_idx])
        X_disc_scaled = scaler.transform(X_disc[:, feat_idx])
        X_val_scaled = scaler.transform(X_val[:, feat_idx])

        clf = make_classifier()
        clf.fit(X_disc_scaled, y_disc)
        check_convergence(clf, f"n_features={n_feat}")

        val_scores = clf.predict_proba(X_val_scaled)[:, 1]
        val_auroc = roc_auc_score(y_val, val_scores)
        nonzero = (clf.coef_ != 0).sum()

        print(f"  n_features={n_feat}: val_AUROC={val_auroc:.4f} | nonzero_weights={nonzero}")
        results[n_feat] = {"clf": clf, "scaler": scaler, "feat_idx": feat_idx, "val_auroc": val_auroc, "nonzero_weights": int(nonzero)}

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_config = n_feat

    print(f"\nBest config by validation AUROC: n_features={best_config}")
    best = results[best_config]

    X_val_scaled = best["scaler"].transform(X_val[:, best["feat_idx"]])
    val_scores = best["clf"].predict_proba(X_val_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)
    threshold = select_threshold_at_min_recall(precisions, recalls, thresholds, min_recall=0.90)

    X_holdout_scaled = best["scaler"].transform(X_holdout[:, best["feat_idx"]])
    holdout_scores = best["clf"].predict_proba(X_holdout_scaled)[:, 1]
    holdout_preds = (holdout_scores >= threshold).astype(int)

    auroc = roc_auc_score(y_holdout, holdout_scores)
    auprc = average_precision_score(y_holdout, holdout_scores)
    cm = confusion_matrix(y_holdout, holdout_preds)
    tn, fp = cm[0]
    fn, tp = cm[1]
    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n--- FROZEN final-holdout results (n_features={best_config}) ---")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

    weight_order = np.argsort(-np.abs(best["clf"].coef_[0]))[:10]
    print("\nTop 10 classifier weights (global feature_idx, weight):")
    for local_pos in weight_order:
        global_feature_id = int(best["feat_idx"][local_pos])
        weight = float(best["clf"].coef_[0, local_pos])
        print(f"  Feature #{global_feature_id}: {weight:.4f}")

    top_weights = [(int(best["feat_idx"][p]), float(best["clf"].coef_[0, p])) for p in weight_order]

    return {
        "best_n_features": int(best_config),
        "selected_features": best["feat_idx"].tolist(),
        "coefficients": best["clf"].coef_[0].tolist(),
        "intercept": float(best["clf"].intercept_[0]),
        "scaler_mean": best["scaler"].mean_.tolist(),
        "scaler_scale": best["scaler"].scale_.tolist(),
        "validation_auroc": float(best["val_auroc"]),
        "threshold": float(threshold),
        "threshold_target_recall": 0.90,
        "holdout_auroc": float(auroc),
        "holdout_auprc": float(auprc),
        "holdout_precision": float(precision),
        "holdout_recall": float(recall),
        "holdout_f1": float(f1),
        "holdout_fpr": float(fpr),
        "confusion_matrix": cm.tolist(),
        "top_weights": top_weights,
        "holdout_offset": 2500,
        "holdout_per_class": 500,
    }

if __name__ == "__main__":
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    discovery_data = torch.load("results/harm_feature_discovery.pt")
    eval_data = torch.load("results/harm_feature_monitor_eval.pt")

    cohens_d = discovery_data["cohens_d"]

    splits = load_beavertails_split(seed=42)
    disc_h, disc_s = splits["discovery"]
    val_h, val_s = splits["validation"]

    assert discovery_data["discovery_harmful_ids"] == [r["id"] for r in disc_h], "Discovery harmful IDs mismatch!"
    assert discovery_data["discovery_safe_ids"] == [r["id"] for r in disc_s], "Discovery safe IDs mismatch!"
    assert eval_data["validation_harmful_ids"] == [r["id"] for r in val_h], "Validation harmful IDs mismatch!"
    assert eval_data["validation_safe_ids"] == [r["id"] for r in val_s], "Validation safe IDs mismatch!"
    print("Split/artifact alignment verified.\n")

    disc_h_acts = discovery_data["discovery_harmful_activations"]
    disc_s_acts = discovery_data["discovery_safe_activations"]
    val_h_acts = eval_data["validation_harmful_activations"]
    val_s_acts = eval_data["validation_safe_activations"]

    print("Running mandatory sanity check before trusting anything else...")
    sanity_check_single_feature(disc_h_acts, disc_s_acts, val_h_acts, val_s_acts, feature_idx=3149)

    holdout_h, holdout_s = get_final_holdout(offset=2500, holdout_n=500)
    print("Computing FINAL HOLDOUT activations...")
    holdout_h_acts = get_per_example_activations(model, sae, holdout_h)
    holdout_s_acts = get_per_example_activations(model, sae, holdout_s)

    results = train_and_evaluate_classifier(disc_h_acts, disc_s_acts, val_h_acts, val_s_acts, holdout_h_acts, holdout_s_acts, cohens_d)

    torch.save(results, "results/multi_feature_classifier.pt")