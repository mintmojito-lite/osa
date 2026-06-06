import os
import torch
import torch.nn as nn

from models.foundation import SleepJEPAEncoder
from models.fusion import MultiModalFusion
from models.bi_mamba import BiMambaLayer
from models.der_head import DERHead

class MambaPedOSA(nn.Module):
    """
    Full ABi-Mamba Architecture with Deep Evidential Regression.
    """
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = {
                'encoder_channels': [1, 64, 128, 256],
                'encoder_strides': [2, 2, 2],
                'ssm_d_model': 256,
                'ssm_d_state': 16,
                'ssm_d_conv': 4,
                'ssm_expand': 2,
                'ssm_n_layers': 6,
                'dropout': 0.1
            }
            
        # 1. Base ECG Encoder
        self.encoder = SleepJEPAEncoder(
            channels=config['encoder_channels'],
            strides=config['encoder_strides'],
            dropout=config['dropout']
        )
        
        # 2. Multi-Modal Fusion
        self.fusion = MultiModalFusion(ssm_d_model=config['ssm_d_model'])
        
        # 3. Stack of ABi-Mamba layers
        self.ssm_layers = nn.ModuleList([
            BiMambaLayer(
                d_model=config['ssm_d_model'],
                d_state=config['ssm_d_state'],
                d_conv=config['ssm_d_conv'],
                expand=config['ssm_expand'],
                p=config['dropout']
            ) for _ in range(config['ssm_n_layers'])
        ])
        
        # 4. Global Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. Evidential Regression Head
        self.der_head = DERHead(in_features=config['ssm_d_model'])
        
    def forward(self, ecg, spo2, ptt, clinical):
        # Ensure correct dimensionality
        if ecg.dim() == 2:
            ecg = ecg.unsqueeze(1)
            
        # Extract features from raw ECG
        ecg_features = self.encoder(ecg) # (B, 256, L)
        
        # Fuse modalities
        x = self.fusion(ecg_features, spo2, ptt, clinical) # (B, D, L)
        
        # Prepare for sequence modeling: Mamba expects (B, L, D)
        x = x.transpose(1, 2)
        
        # Process through SSM layers
        for layer in self.ssm_layers:
            x = layer(x) # (B, L, D)
            
        # Prepare for pooling: AdaptiveAvgPool1d expects (B, D, L)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1) # (B, D)
        
        # Regression head
        return self.der_head(x)

def report_parameters(model):
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
    total_params = count_params(model)
    encoder_params = count_params(model.encoder)
    ssm_params = count_params(model.ssm_layers)
    der_params = count_params(model.der_head)
    fusion_params = count_params(model.fusion)
    
    report = (
        f"--- Mamba-PedOSA Architecture Summary ---\n"
        f"Total Parameters:      {total_params:,}\n"
        f"Encoder Parameters:    {encoder_params:,}\n"
        f"Fusion Parameters:     {fusion_params:,}\n"
        f"SSM Stack Parameters:  {ssm_params:,}\n"
        f"DER Head Parameters:   {der_params:,}\n"
        f"-----------------------------------------\n"
    )
    print(report)
    
    # Assert total parameters are less than 6 Million (FP32 budget constraint)
    assert total_params < 6_000_000, f"Total parameters ({total_params:,}) exceed the 6M target!"
    
    return report

if __name__ == "__main__":
    # Change current working directory to project root for correct saving
    import sys
    from pathlib import Path
    
    # Allow running directly from models/mamba_pedosa.py
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    model = MambaPedOSA()
    report = report_parameters(model)
    
    # Save report
    out_dir = project_root / 'models'
    out_path = out_dir / 'architecture_summary.txt'
    
    with open(out_path, 'w') as f:
        f.write(report)
        f.write("\nModel Structure:\n")
        f.write(str(model))
        
    print(f"Saved architecture summary to {out_path}")
