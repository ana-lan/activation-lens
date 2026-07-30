import torch
import torch.nn as nn
import time
from model.sae import SparseAutoencoder, compute_loss

torch.manual_seed(42)

def resample_dead_features(sae, dead_mask, input_dim):
    # dead_mask is a boolean tensor of shape [expansion_factor], True = dead
    num_dead = dead_mask.sum().item()
    if num_dead == 0:
        return 0

    with torch.no_grad():
        # 1. generate fresh random weights for just the dead rows of W_enc
        new_weights = torch.randn(num_dead, input_dim) * 0.01

        # 2. assign them into W_enc only at the dead feature positions
        sae.W_enc[dead_mask] = new_weights

        # 3. reset their biases to zero too
        sae.b_enc[dead_mask] = 0.0

    return num_dead

def train_sae(
    cache_path: str = "data/activation_cache/full_precision/cache.pt",
    input_dim: int = 768,
    expansion_factor: int = 8192,
    batch_size: int = 4096,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    l1_lambda: float = 1e-3,
    device: str = "cpu",
    save_path: str = "model/sae_weights.pt",
    resample_every: int = 5
):
    activations = torch.load(cache_path).float().to(device)
    print(f"Activations stats: mean={activations.mean().item():.6f}, std={activations.std().item():.6f}")

    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    num_samples = activations.shape[0]
    num_batches = num_samples // batch_size

    # tracks whether each feature has fired at all since the last resampling check
    feature_ever_active = torch.zeros(expansion_factor, dtype=torch.bool, device=device)

    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples)
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0

        start_time = time.time()

        for b in range(num_batches):
            batch_indices = perm[b * batch_size : (b + 1) * batch_size]
            batch = activations[batch_indices]

            features, reconstruction = sae(batch)
            loss, recon_loss, sparsity_loss = compute_loss(batch, features, reconstruction, l1_lambda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            # 4. update our tracker: which features fired at all in this batch
            batch_active = (features > 0).any(dim=0)
            feature_ever_active |= batch_active

            epoch_recon_loss += recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | recon_loss: {epoch_recon_loss/num_batches:.4f} | sparsity_loss: {epoch_sparsity_loss/num_batches:.4f} | time: {elapsed:.1f}s")

        # 5. every `resample_every` epochs, resample dead features and reset the tracker
        if (epoch + 1) % resample_every == 0:
            dead_mask = ~feature_ever_active
            num_resampled = resample_dead_features(sae, dead_mask, input_dim)
            print(f"  -> Resampled {num_resampled} dead features")
            feature_ever_active = torch.zeros(expansion_factor, dtype=torch.bool, device=device)  # reset tracker

    torch.save(sae.state_dict(), save_path)
    print(f"Saved trained SAE to {save_path}")

if __name__ == "__main__":
    train_sae()