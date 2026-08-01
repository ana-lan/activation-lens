import torch
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

_eval_data = torch.load("results/harm_feature_monitor_eval.pt")
FROZEN_FEATURE = _eval_data["frozen_feature"]

def monitor_generation(model, sae, prompt: str, live_threshold: float, max_new_tokens: int = 30, layer: int = 6):
    """Generates text token-by-token. For each generated token, the activation
    is computed AFTER that token is appended, so the score genuinely belongs
    to the token it's logged next to (fixes the earlier off-by-one bug)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"blocks.{layer}.hook_resid_post"

    per_token_log = []
    flagged = False
    flagged_at_token = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            # re-run on the updated sequence so the activation genuinely belongs
            # to the just-generated token, not the previous one
            _, updated_cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            token_activation = updated_cache[hook_name][0, -1]
            feature_score = sae(token_activation.unsqueeze(0))[0][0, FROZEN_FEATURE].item()

        token_str = model.to_string(next_token[0])
        is_flagged = feature_score >= live_threshold

        per_token_log.append({"step": step, "token": token_str, "feature_score": feature_score, "flagged": is_flagged})

        if is_flagged and not flagged:
            flagged, flagged_at_token = True, step

        if next_token.item() == model.tokenizer.eos_token_id:
            break

    return {
        "generated_text": model.to_string(tokens[0]),
        "per_token_log": per_token_log,
        "flagged": flagged,
        "flagged_at_token": flagged_at_token,
        "threshold": live_threshold,
        "feature_used": FROZEN_FEATURE,
        "disclaimer": "Experimental single-feature signal. Not a validated safety monitor.",
    }

if __name__ == "__main__":
    model = load_base_model(device="cpu")
    sae = load_trained_sae("model/sae_weights.pt").to("cpu")
    sae.eval()

    live_calib = torch.load("results/harm_feature_live_calibration.pt")
    live_threshold = live_calib["live_threshold"]

    result = monitor_generation(model, sae, prompt="The best way to deal with a difficult situation is", live_threshold=live_threshold)
    print(f"Generated: {result['generated_text']}")
    print(f"Flagged: {result['flagged']} (at token {result['flagged_at_token']})")
    for entry in result["per_token_log"][:10]:
        print(f"  Step {entry['step']}: '{entry['token']}' | score={entry['feature_score']:.4f} | flagged={entry['flagged']}")