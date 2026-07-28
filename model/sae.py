import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 768, expansion_factor: int = 8192):
        super().__init__()
        # 1. define the encoder weight matrix and bias
        self.W_enc = nn.Parameter(torch.randn(expansion_factor, input_dim) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(expansion_factor))

        # 2. define the decoder weight matrix and bias
        self.W_dec = nn.Parameter(torch.randn(input_dim, expansion_factor) * 0.01) 
        self.b_dec = nn.Parameter(torch.zeros(input_dim)) 

    def forward(self, x):
        # 3. encoder: linear transform + ReLU
        features = torch.relu(x @ self.W_enc.T + self.b_enc)

        # 4. decoder: linear transform back down
        reconstruction = features @ self.W_dec.T + self.b_dec

        return features, reconstruction

def compute_loss(x, features, reconstruction, l1_lambda: float = 1e-3):
    # 5. reconstruction error: mean squared difference
    recon_loss = torch.mean((x - reconstruction) ** 2) 

    # 6. sparsity penalty: sum of active feature magnitudes
    sparsity_loss = torch.mean(torch.sum(torch.abs(features), dim=-1)) 

    total_loss = recon_loss + l1_lambda * sparsity_loss
    return total_loss, recon_loss, sparsity_loss