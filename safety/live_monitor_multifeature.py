import torch
import numpy as np
from scipy.special import expit
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

_clf_data = torch.load("results/multi_feature_classifier.pt")
_live_calib = torch.load("results/live_multifeature_calibration.pt")

SELECTED_FEATURES = _clf_data["selected_features"]
COEFFICIENTS = np.array(_clf_data["coefficients"])
INTERCEPT = _clf_data["intercept"]
SCALER_MEAN = np.array(_clf_data["scaler_mean"])
SCALER_SCALE = np.array(_clf_data["scaler_scale"])

MIN_PREFIX_TOKENS = _live_calib["min_prefix_tokens"]
LIVE_THRESHOLD = _live_calib["live_threshold"]
SHORT_RESPONSE_POLICY = _live_calib["short_response_policy"]

if SHORT_RESPONSE_POLICY != "score_at_end":
    raise ValueError(f"Unsupported short-response policy: {SHORT_RESPONSE_POLICY}")

def score_running_mean(running_mean_selected_features):
    scaled = (running_mean_selected_features - SCALER_MEAN) / SCALER_SCALE
    logit = np.dot(scaled, COEFFICIENTS) + INTERCEPT
    return float(expit(logit))

def monitor_generation_multifeature(model, sae, prompt: str, max_new_tokens: int = 30, layer: int = 6):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"blocks.{layer}.hook_resid_post"

    running_sum = np.zeros(len(SELECTED_FEATURES))
    num_tokens_seen = 0

    per_token_log = []
    flagged = False
    flagged_at_token = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            _, updated_cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            token_activation = updated_cache[hook_name][0, -1]
            features, _ = sae(token_activation.unsqueeze(0))
            selected_values = features[0, SELECTED_FEATURES].numpy()

        running_sum += selected_values
        num_tokens_seen += 1
        running_mean = running_sum / num_tokens_seen
        score = score_running_mean(running_mean)

        token_id = next_token.item()
        is_eos = token_id == model.tokenizer.eos_token_id
        is_final_step = step == max_new_tokens - 1

        reached_minimum = num_tokens_seen >= MIN_PREFIX_TOKENS
        short_response_ended = (
            num_tokens_seen < MIN_PREFIX_TOKENS
            and (is_eos or is_final_step)
            and SHORT_RESPONSE_POLICY == "score_at_end"
        )

        eligible_for_alert = reached_minimum or short_response_ended

        above_threshold = eligible_for_alert and score >= LIVE_THRESHOLD

        if above_threshold and not flagged:
            flagged, flagged_at_token = True, step

        token_str = model.to_string(next_token[0])
        per_token_log.append({
            "step": step, "token_id": token_id, "token": token_str,
            "score": float(score), "eligible_for_alert": bool(eligible_for_alert),
            "above_threshold": bool(above_threshold),
            "alert_active": bool(flagged),
        })

        if is_eos:
            break

    return {
        "generated_text": model.to_string(tokens[0]),
        "per_token_log": per_token_log,
        "flagged": flagged,
        "flagged_at_token": flagged_at_token,
        "threshold": LIVE_THRESHOLD,
        "min_prefix_tokens": MIN_PREFIX_TOKENS,
        "short_response_policy": SHORT_RESPONSE_POLICY,
        "num_features_used": len(SELECTED_FEATURES),
        "disclaimer": "100-feature regularized linear probe, live prefix-calibrated: held-out AUROC 0.758, FPR 0.652 at 90%+ recall operating point. Prototype, not production-grade. Calibrated on standalone responses; real-time generation is prompt-conditioned, a noted distribution shift.",
    }

if __name__ == "__main__":
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    result = monitor_generation_multifeature(model, sae, prompt="The best way to deal with a difficult situation is")
    print(f"Generated: {result['generated_text']}")
    print(f"Flagged: {result['flagged']} (at token {result['flagged_at_token']})")
    for entry in result["per_token_log"][:10]:
        print(f"  Step {entry['step']}: '{entry['token']}' | score={entry['score']:.4f} | eligible={entry['eligible_for_alert']} | above_threshold={entry['above_threshold']} | alert_active={entry['alert_active']}")