import torch
from model.sae import SparseAutoencoder

def load_trained_sae(sae_weights_path: str, input_dim: int = 768, expansion_factor: int = 8192):
    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor)
    sae.load_state_dict(torch.load(sae_weights_path))
    sae.eval()
    return sae

def evaluate_sparsity(sae, activations_sample):
    with torch.no_grad():
        features, reconstruction = sae(activations_sample)

    nonzero_frac = (features > 0).float().mean()
    recon_error = torch.mean((activations_sample - reconstruction) ** 2)

    return nonzero_frac.item(), recon_error.item()

if __name__ == "__main__":
    sae = load_trained_sae("model/sae_weights.pt")
    activations = torch.load("data/activation_cache/full_precision/cache.pt").float()
    sample = activations[:5000]

    nonzero_frac, recon_error = evaluate_sparsity(sae, sample)
    print(f"Post-training nonzero_frac: {nonzero_frac:.4f}")
    print(f"Post-training reconstruction error: {recon_error:.4f}")