import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logging.warning("mamba_ssm not found or failed to import. Using SimpleSSM fallback.")

class SimpleSSM(nn.Module):
    """
    A simplified selective state space model fallback using Zero-Order Hold (ZOH) discretization.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Learnable parameters (A must be strictly negative for stability)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Selective projections
        self.proj_delta = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        
    def forward(self, x):
        """
        x: (B, L, D)
        """
        B_size, L, D = x.shape
        N = self.d_state
        
        # Projections
        delta = F.softplus(self.proj_delta(x)) # (B, L, D)
        B_val = self.proj_B(x) # (B, L, N)
        C_val = self.proj_C(x) # (B, L, N)
        
        h = torch.zeros(B_size, D, N, device=x.device)
        y = torch.zeros(B_size, L, D, device=x.device)
        
        # Simple RNN-style loop over sequence length
        for t in range(L):
            xt = x[:, t, :] # (B, D)
            dt = delta[:, t, :].unsqueeze(2) # (B, D, 1)
            bt = B_val[:, t, :].unsqueeze(1) # (B, 1, N)
            ct = C_val[:, t, :].unsqueeze(1) # (B, 1, N)
            
            # ZOH Discretization (A is strictly negative)
            A_stable = -torch.exp(self.A_log)
            A_bar = torch.exp(dt * A_stable) # (B, D, N)
            B_bar = dt * bt # Simplified discretization for B
            
            # Update state
            h = A_bar * h + B_bar * xt.unsqueeze(2) # (B, D, N)
            
            # Output
            yt = (h * ct).sum(dim=2) + self.D * xt # (B, D)
            y[:, t, :] = yt
            
        return y

class MambaBlock(nn.Module):
    """
    Mamba sequence modeling block with pre-norm and residual connection.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, p=0.1):
        super().__init__()
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.mamba = SimpleSSM(d_model=d_model, d_state=d_state)
            
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        """
        x: (B, L, D)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual
