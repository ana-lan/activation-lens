import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def plot_score_distributions(frozen_feature: int, val_harmful, val_safe, save_path: str = "results/score_distribution.png"):
    import matplotlib.pyplot as plt

    h_scores = val_harmful[:, frozen_feature].numpy()
    s_scores = val_safe[:, frozen_feature].numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(h_scores, bins=40, alpha=0.5, label="Harmful", color="red")
    plt.hist(s_scores, bins=40, alpha=0.5, label="Safe", color="green")
    plt.xlabel(f"Feature #{frozen_feature} activation")
    plt.ylabel("Count")
    plt.title(f"Feature #{frozen_feature} — Harmful vs Safe activation distribution (validation set)")
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved distribution plot to {save_path}")

def diagnostic_check_all_candidates(top_indices, val_harmful, val_safe, test_harmful, test_safe):
    """Measures AUROC/AUPRC for ALL top-10 discovery candidates on val+test —
    diagnostic only, does NOT reselect a 'better' feature."""
    print("\nDiagnostic: how do ALL top-10 discovery candidates perform on held-out data?")
    print("(measurement only — the frozen candidate for the project remains feature #3149 / whatever was frozen)\n")

    for idx in top_indices:
        idx = idx.item() if torch.is_tensor(idx) else idx

        val_scores = torch.cat([val_harmful[:, idx], val_safe[:, idx]]).numpy()
        val_labels = torch.cat([torch.ones(val_harmful.shape[0]), torch.zeros(val_safe.shape[0])]).numpy()

        test_scores = torch.cat([test_harmful[:, idx], test_safe[:, idx]]).numpy()
        test_labels = torch.cat([torch.ones(test_harmful.shape[0]), torch.zeros(test_safe.shape[0])]).numpy()

        val_auroc = roc_auc_score(val_labels, val_scores)
        test_auroc = roc_auc_score(test_labels, test_scores)
        test_auprc = average_precision_score(test_labels, test_scores)

        print(f"  Feature #{idx}: val_AUROC={val_auroc:.4f} | test_AUROC={test_auroc:.4f} | test_AUPRC={test_auprc:.4f}")

if __name__ == "__main__":
    discovery_data = torch.load("results/harm_feature_discovery.pt")
    eval_data = torch.load("results/harm_feature_monitor_eval.pt")

    frozen_feature = eval_data["frozen_feature"]
    val_harmful = eval_data["validation_harmful_activations"]
    val_safe = eval_data["validation_safe_activations"]
    test_harmful = eval_data["test_harmful_activations"]
    test_safe = eval_data["test_safe_activations"]
    top_indices = discovery_data["top_indices"]

    plot_score_distributions(frozen_feature, val_harmful, val_safe)
    diagnostic_check_all_candidates(top_indices, val_harmful, val_safe, test_harmful, test_safe)