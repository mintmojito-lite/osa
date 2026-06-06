import torch
import torch.nn as nn

class SENet(nn.Module):
    """
    Squeeze-and-Excitation Network modified for sequence data.
    Takes concatenated bidirectional features (B, L, 2D),
    applies channel attention via global average pooling,
    and projects back to (B, L, D).
    """
    def __init__(self, channels):
        super().__init__()
        # channels is 2*D
        reduction_channels = max(1, channels // 8)
        
        self.squeeze = nn.Sequential(
            nn.Linear(channels, reduction_channels),
            nn.GELU(),
            nn.Linear(reduction_channels, channels),
            nn.Sigmoid()
        )
        
        self.project = nn.Linear(channels, channels // 2)
        
    def forward(self, x):
        """
        x: (B, L, 2*D)
        """
        # Global average pool over L dimension -> (B, 2*D)
        pooled = x.mean(dim=1)
        
        # Calculate channel attention scale -> (B, 2*D)
        scale = self.squeeze(pooled)
        
        # Unsqueeze for broadcasting over L -> (B, 1, 2*D)
        scale = scale.unsqueeze(1)
        
        # Apply attention
        x = x * scale
        
        # Project back to D -> (B, L, D)
        out = self.project(x)
        return out
