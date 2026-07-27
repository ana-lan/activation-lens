import torch
from transformer_lens import HookedTransformer
from dotenv import load_dotenv
load_dotenv()

def load_base_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(
        "gpt2",
        device=device
    )

    return model

if __name__ == "__main__":
    model = load_base_model()
    print(model.cfg)