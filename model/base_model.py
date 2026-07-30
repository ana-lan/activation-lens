import torch
from transformer_lens import HookedTransformer

def load_base_model(device: str = "cpu"):
    return HookedTransformer.from_pretrained("gpt2", device=device)

if __name__ == "__main__":
    model = load_base_model()
    print(model.cfg)