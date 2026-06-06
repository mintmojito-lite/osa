import torch
import torch.nn as nn
import logging

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    logging.warning("mamba_ssm not found or failed to import. Using a fallback GRU-based mock for testing.")
    MAMBA_AVAILABLE = False

class MambaMock(nn.Module):
    """
    A fallback mock module using GRU that matches the interface of Mamba.
    Used for testing the pipeline when mamba_ssm is not installed/compilable.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        # We use a GRU as a sequence modeling fallback
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        
    def forward(self, x):
        # x is (B, L, D)
        out, _ = self.gru(x)
        return out

class MambaSequenceModel(nn.Module):
    """
    Sequence modeling block utilizing Mamba for the fused temporal features.
    """
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, n_layers=6):
        super().__init__()
        
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_layers):
            if MAMBA_AVAILABLE:
                self.layers.append(
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                )
            else:
                self.layers.append(
                    MambaMock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                )
            self.norms.append(nn.LayerNorm(d_model))
            
    def forward(self, x):
        """
        Args:
            x: (B, D, L) - note that Mamba expects (B, L, D)
        Returns:
            out: (B, D, L)
        """
        # Convert (B, D, L) to (B, L, D)
        x = x.transpose(1, 2)
        
        for layer, norm in zip(self.layers, self.norms):
            # Pre-norm architecture
            residual = x
            x = norm(x)
            x = layer(x)
            x = x + residual
            
        # Convert back to (B, D, L)
        x = x.transpose(1, 2)
        return x
