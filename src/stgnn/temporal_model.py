import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    """
    Temporal processing using GRU (Gated Recurrent Unit) for processing graph embeddings over time.
    """
    def __init__(self, in_features, hidden_size, num_layers=2):
        super(TemporalModel, self).__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (Batch, Timesteps, Nodes * Features)
        out, h_n = self.gru(x)
        
        # Take the output of the last timestep
        last_out = out[:, -1, :] # (Batch, hidden_size)
        
        return self.layer_norm(last_out)
