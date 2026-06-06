import torch
import torch.nn as nn
from .mamba_block import MambaBlock
from .senet import SENet

class BiMambaLayer(nn.Module):
    """
    Bidirectional Mamba layer with Attention-based Fusion (ABi-Mamba).
    Processes the sequence in both forward and backward directions,
    then fuses the representations using SENet.
    """
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, p=0.1):
        super().__init__()
        
        self.forward_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            p=p
        )
        
        self.backward_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            p=p
        )
        
        self.senet = SENet(channels=d_model * 2)
        
    def forward(self, x):
        """
        x: (B, L, D)
        """
        # Forward processing
        fwd = self.forward_mamba(x)
        
        # Backward processing
        # Flip sequence along length dimension (dim=1), process, then flip back
        x_flipped = torch.flip(x, dims=[1])
        bwd = self.backward_mamba(x_flipped)
        bwd = torch.flip(bwd, dims=[1])
        
        # Concatenate forward and backward features -> (B, L, 2*D)
        combined = torch.cat([fwd, bwd], dim=-1)
        
        # Apply Channel Attention (SENet) -> (B, L, D)
        out = self.senet(combined)
        return out
