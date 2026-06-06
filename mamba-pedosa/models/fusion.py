import torch
import torch.nn as nn
from einops import repeat

class MultiModalFusion(nn.Module):
    """
    Fuses ECG latent features, SpO2 sequence, PTT scalar, and Clinical priors 
    into a unified representation for the Mamba backbone.
    """
    def __init__(self, ssm_d_model=256):
        super().__init__()
        
        # SpO2 branch
        self.spo2_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool1d(960)
        )
        
        # PTT scalar expansion
        self.ptt_expand = nn.Linear(1, 960)
        
        # Clinical prior embedding
        self.clinical_embed = nn.Linear(3, 32)
        
        # Final projection to SSM dimension
        # Input to this conv is: ECG(256) + SpO2(128) + PTT(1) + Clinical(32) = 417
        self.projection = nn.Conv1d(417, ssm_d_model, kernel_size=1)
        
    def forward(self, ecg_features, spo2, ptt, clinical):
        """
        Args:
            ecg_features: (B, 256, 960)
            spo2: (B, 1, 960) or (B, 960)
            ptt: (B, 1) or (B,)
            clinical: (B, 3)
            
        Returns:
            fused_features: (B, ssm_d_model, 960)
        """
        B = ecg_features.size(0)
        
        # Ensure correct SpO2 dimensions
        if spo2.dim() == 2:
            spo2 = spo2.unsqueeze(1) # (B, 1, 960)
            
        # Ensure correct PTT dimensions
        if ptt.dim() == 1:
            ptt = ptt.unsqueeze(1) # (B, 1)
            
        # Process SpO2
        spo2_feat = self.spo2_branch(spo2) # (B, 128, 960)
        
        # Process PTT
        ptt_feat = self.ptt_expand(ptt) # (B, 960)
        ptt_feat = ptt_feat.unsqueeze(1) # (B, 1, 960)
        
        # Process Clinical
        clin_embed = self.clinical_embed(clinical) # (B, 32)
        # Repeat along sequence length
        clin_feat = repeat(clin_embed, 'b c -> b c l', l=960) # (B, 32, 960)
        
        # Concatenate all features along channel dimension
        # ecg_features: (B, 256, 960)
        fused_concat = torch.cat([ecg_features, spo2_feat, ptt_feat, clin_feat], dim=1) # (B, 417, 960)
        
        # Project to SSM input dim
        fused_proj = self.projection(fused_concat) # (B, ssm_d_model, 960)
        
        return fused_proj
