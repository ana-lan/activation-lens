import torch
from model.base_model import load_base_model

def get_activations(model, text: str, layer: int = 6):
    tokens = model.to_tokens(text, prepend_bos=True, move_to_device=True)

    logits, cache = model.run_with_cache(tokens)

    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = cache[hook_name]

    return activations

if __name__ == "__main__":
    model = load_base_model()
    acts = get_activations(model, "The Golden Gate Bridge is in San Francisco.")
    print(acts.shape)