import os
import torch
import torch.nn as nn

from models.foundation import SleepJEPAEncoder
from models.fusion import MultiModalFusion
from models.bi_mamba import BiMambaLayer

class EDLClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, num_classes),
            nn.Softplus() # Ensures non-negative evidence e_k >= 0
        )
        
    def forward(self, x):
        evidence = self.mlp(x)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        uncertainty = 4.0 / S # K=4 classes
        return {
            'evidence': evidence,
            'alpha': alpha,
            'S': S,
            'prob': prob,
            'uncertainty': uncertainty
        }

class AHIRegressionHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features // 2, 1)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class MambaPedOSA(nn.Module):
    """
    Full ABi-Mamba Architecture with Evidential Classification, AHI Regression, and Sleep/Wake Staging.
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
        
        # 5. Output Heads (Classification, Regression, Sleep/Wake)
        self.edl_head = EDLClassificationHead(in_features=config['ssm_d_model'])
        self.ahi_head = AHIRegressionHead(in_features=config['ssm_d_model'])
        
        # Binary sleep/wake classifier
        self.sleep_wake_head = nn.Linear(config['ssm_d_model'], 1)
        
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
        
        # Regression head & Aux Classification
        out = self.edl_head(x)
        out['ahi_pred'] = self.ahi_head(x).squeeze(-1)
        out['sleep_wake_logits'] = self.sleep_wake_head(x).squeeze(-1)
        return out

def report_parameters(model):
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
    total_params = count_params(model)
    encoder_params = count_params(model.encoder)
    ssm_params = count_params(model.ssm_layers)
    edl_params = count_params(model.edl_head)
    ahi_params = count_params(model.ahi_head)
    fusion_params = count_params(model.fusion)
    
    report = (
        f"--- Mamba-PedOSA Architecture Summary ---\n"
        f"Total Parameters:      {total_params:,}\n"
        f"Encoder Parameters:    {encoder_params:,}\n"
        f"Fusion Parameters:     {fusion_params:,}\n"
        f"SSM Stack Parameters:  {ssm_params:,}\n"
        f"EDL Head Parameters:   {edl_params:,}\n"
        f"AHI Head Parameters:   {ahi_params:,}\n"
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
