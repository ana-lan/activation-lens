import torch
from datasets import load_dataset
from model.base_model import load_base_model
from model.evaluate_sae import load_trained_sae

def find_top_activating_examples_with_model(
    model,
    sae,
    feature_idx: int,
    num_texts: int = 200,
    layer: int = 6,
    top_k: int = 10
):
    dataset = load_dataset("Skylion007/openwebtext", split=f"train[:{num_texts}]")

    results = []

    for example in dataset:
        text = example["text"]
        if not text.strip():
            continue

        tokens = model.to_tokens(text, prepend_bos=True)
        str_tokens = model.to_str_tokens(text, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            acts = cache[f"blocks.{layer}.hook_resid_post"].squeeze(0)
            features, _ = sae(acts)

        feature_values = features[:, feature_idx]
        max_val, max_pos = torch.max(feature_values, dim=0)

        start = max(0, max_pos.item() - 5)
        end = min(len(str_tokens), max_pos.item() + 5)
        context = "".join(str_tokens[start:end])

        results.append((max_val.item(), context))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def find_top_activating_examples(
    feature_idx: int,
    num_texts: int = 200,
    layer: int = 6,
    top_k: int = 10,
    sae_weights_path: str = "model/sae_weights.pt"
):
    device = "cpu"
    model = load_base_model()
    model = model.to(device)

    sae = load_trained_sae(sae_weights_path)
    sae = sae.to(device)

    return find_top_activating_examples_with_model(model, sae, feature_idx, num_texts, layer, top_k)

if __name__ == "__main__":
    top_examples = find_top_activating_examples(feature_idx=0)
    for val, context in top_examples:
        print(f"Activation: {val:.3f} | Context: {context}")