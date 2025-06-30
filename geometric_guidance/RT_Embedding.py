import torch
import torch.nn as nn

class RTEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, RT_size=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(RT_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.RT_size = RT_size

    def forward(self, RT):
        rt_emb = self.mlp(RT)
        return rt_emb