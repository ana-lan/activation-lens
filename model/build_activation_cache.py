import torch
import os
import time
from datasets import load_dataset
from model.base_model import load_base_model

def build_cache(num_samples: int = 2500, layer: int = 6, save_path: str = "data/activation_cache/full_precision/cache.pt", batch_size: int = 16, checkpoint_every: int = 200):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # force CPU explicitly, sidestepping MPS memory issues entirely
    device = "cpu"
    model = load_base_model()
    model = model.to(device)

    dataset = load_dataset("Skylion007/openwebtext", split="train[:3000]")
    texts = [ex["text"] for ex in dataset.select(range(num_samples)) if ex["text"].strip()]

    all_vectors = []
    start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        for text in batch_texts:
            tokens = model.to_tokens(text, prepend_bos=True, move_to_device=True)
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens)
            acts = cache[f"blocks.{layer}.hook_resid_post"].squeeze(0)
            all_vectors.append(acts)

        if i % (checkpoint_every) == 0 and i > 0:
            elapsed = time.time() - start_time
            checkpoint_dataset = torch.cat(all_vectors, dim=0).half()
            torch.save(checkpoint_dataset, save_path)
            print(f"Checkpoint at {i} samples ({checkpoint_dataset.shape[0]} vectors), {elapsed:.1f}s elapsed")

    full_dataset = torch.cat(all_vectors, dim=0).half()
    torch.save(full_dataset, save_path)
    elapsed = time.time() - start_time
    print(f"Done. {full_dataset.shape[0]} vectors saved in {elapsed:.1f}s")

    # sanity check before finishing
    print(f"Sanity check - mean: {full_dataset.float().mean().item():.6f}, std: {full_dataset.float().std().item():.6f}")

if __name__ == "__main__":
    build_cache()