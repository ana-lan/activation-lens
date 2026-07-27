import torch
import os
import time
from datasets import load_dataset
from model.base_model import load_base_model
from model.hooks import get_activations

def build_cache(num_samples: int = 2500, layer: int = 6, save_path: str = "data/activation_cache/full_precision/cache.pt", checkpoint_every: int = 100, resume_from: int = 0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = load_base_model()
    dataset = load_dataset("Skylion007/openwebtext", split="train[:3000]")

    if resume_from > 0 and os.path.exists(save_path):
        existing = torch.load(save_path)
        all_vectors = [existing]
        print(f"Resumed from checkpoint: {existing.shape[0]} vectors already loaded")
    else:
        all_vectors = []

    start_time = time.time()

    for i, example in enumerate(dataset):
        if i < resume_from:
            continue
        if i >= num_samples:
            break

        text = example["text"]
        if not text.strip():
            continue

        acts = get_activations(model, text, layer=layer)
        flattened = acts.squeeze(0)
        all_vectors.append(flattened)

        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i}/{num_samples} ({elapsed:.1f}s elapsed)")

        if i > 0 and i % checkpoint_every == 0:
            checkpoint_dataset = torch.cat(all_vectors, dim=0).half()
            torch.save(checkpoint_dataset, save_path)
            print(f"Checkpoint saved at {i} samples ({checkpoint_dataset.shape[0]} vectors) -> {save_path}")

    full_dataset = torch.cat(all_vectors, dim=0).half()
    torch.save(full_dataset, save_path)
    elapsed = time.time() - start_time
    print(f"Done. Processed up to {num_samples} samples in {elapsed:.1f} seconds")
    print(f"Saved {full_dataset.shape[0]} activation vectors to {save_path}")
    os._exit(0)

if __name__ == "__main__":
    build_cache()