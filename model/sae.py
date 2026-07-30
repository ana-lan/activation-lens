import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 768, expansion_factor: int = 8192):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(expansion_factor, input_dim) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(expansion_factor))

        self.W_dec = nn.Parameter(torch.randn(input_dim, expansion_factor) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        features = torch.relu(x @ self.W_enc.T + self.b_enc)
        reconstruction = features @ self.W_dec.T + self.b_dec
        return features, reconstruction

    def normalize_decoder(self):
        """Force each feature's decoder weight vector to unit norm.
        Call this after every optimizer step during training."""
        with torch.no_grad():
            # W_dec has shape [input_dim, expansion_factor] — each COLUMN is one feature's decoder vector
            norms = self.W_dec.norm(dim=0, keepdim=True)  # shape [1, expansion_factor]
            self.W_dec.data = self.W_dec.data / (norms + 1e-8)  # small epsilon avoids divide-by-zero

def compute_loss(x, features, reconstruction, l1_lambda: float = 1e-3):
    recon_loss = torch.mean((x - reconstruction) ** 2)
    sparsity_loss = torch.mean(torch.sum(torch.abs(features), dim=-1))
    total_loss = recon_loss + l1_lambda * sparsity_loss
    return total_loss, recon_loss, sparsity_loss