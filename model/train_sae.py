import torch
import torch.nn as nn
import time
from model.sae import SparseAutoencoder, compute_loss

def train_sae(
    cache_path: str = "data/activation_cache/full_precision/cache.pt",
    input_dim: int = 768,
    expansion_factor: int = 8192,
    batch_size: int = 4096,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    l1_lambda: float = 1e-3,
    device: str = "mps",
    save_path: str = "model/sae_weights.pt"
):
    # 1. load the activation cache
    activations = torch.load(cache_path).float().to(device)  # convert from float16 back to float32 for training stability
    print(f"Activations stats: mean={activations.mean().item():.6f}, std={activations.std().item():.6f}, min={activations.min().item():.6f}, max={activations.max().item():.6f}")

    # 2. create the SAE and move it to device
    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor).to(device)

    # 3. create the optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    num_samples = activations.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(num_epochs):
        # shuffle the data at the start of each epoch
        perm = torch.randperm(num_samples)
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0

        start_time = time.time()

        for b in range(num_batches):
            batch_indices = perm[b * batch_size : (b + 1) * batch_size]
            batch = activations[batch_indices]

            # 4. forward pass
            features, reconstruction = sae(batch)

            # 5. compute loss
            loss, recon_loss, sparsity_loss = compute_loss(batch, features, reconstruction, l1_lambda)

            if epoch == 0 and b == 0:
                print(f"Batch stats: mean={batch.mean().item():.6f}, std={batch.std().item():.6f}")
                print(f"Features stats: mean={features.mean().item():.6f}, nonzero_frac={(features > 0).float().mean().item():.6f}")
                print(f"Recon loss (raw): {recon_loss.item():.8f}")
                print(f"Sparsity loss (raw): {sparsity_loss.item():.8f}")

            # 6. backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | recon_loss: {epoch_recon_loss/num_batches:.4f} | sparsity_loss: {epoch_sparsity_loss/num_batches:.4f} | time: {elapsed:.1f}s")

    torch.save(sae.state_dict(), save_path)
    print(f"Saved trained SAE to {save_path}")

if __name__ == "__main__":
    train_sae()