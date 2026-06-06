import torch
import torch.nn as nn
from .foundation import SleepJEPAEncoder, load_sleepjepa_weights
from .fusion import MultiModalFusion
from .mamba import MambaSequenceModel
from .der import DERHead

class MambaPedOSA(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        # Default config values matching model_config.yaml
        if config is None:
            config = {
                'encoder_channels': [1, 64, 128, 256],
                'encoder_strides': [2, 2, 2],
                'ssm_d_model': 256,
                'ssm_d_state': 16,
                'ssm_d_conv': 4,
                'ssm_expand': 2,
                'ssm_n_layers': 6,
                'der_hidden': 512,
                'dropout': 0.1
            }
            
        # 1. Base ECG Encoder (SleepJEPA compatible)
        self.encoder = SleepJEPAEncoder(
            channels=config['encoder_channels'],
            strides=config['encoder_strides'],
            dropout=config['dropout']
        )
        
        # 2. MultiModal Fusion Module
        self.fusion = MultiModalFusion(ssm_d_model=config['ssm_d_model'])
        
        # 3. Mamba Sequence Modeler
        self.mamba = MambaSequenceModel(
            d_model=config['ssm_d_model'],
            d_state=config['ssm_d_state'],
            d_conv=config['ssm_d_conv'],
            expand=config['ssm_expand'],
            n_layers=config['ssm_n_layers']
        )
        
        # 4. Global Pooling & Evidential Regression Head
        self.head = DERHead(
            in_features=config['ssm_d_model'],
            hidden_dim=config['der_hidden'],
            dropout=config['dropout'],
            num_classes=4
        )
        
    def forward(self, ecg, spo2, ptt_swing, clinical):
        """
        Forward pass
        ecg: (B, 1, 7680)
        spo2: (B, 1, 960)
        ptt_swing: (B, 1) or (B,)
        clinical: (B, 3)
        """
        # Ensure channel dim exists for ECG
        if ecg.dim() == 2:
            ecg = ecg.unsqueeze(1)
            
        # 1. ECG Features: (B, 256, 960)
        ecg_feat = self.encoder(ecg)
        
        # 2. Fuse Modalities: (B, ssm_d_model, 960)
        fused = self.fusion(ecg_feat, spo2, ptt_swing, clinical)
        
        # 3. Temporal Modeling: (B, ssm_d_model, 960)
        mamba_out = self.mamba(fused)
        
        # Global Average Pooling: (B, ssm_d_model)
        pooled = mamba_out.mean(dim=2)
        
        # 4. Regression & Classification Head
        mu, v, alpha, beta, class_logits = self.head(pooled)
        
        return {
            'mu': mu,
            'v': v,
            'alpha': alpha,
            'beta': beta,
            'class_logits': class_logits
        }
        
def build_model(config, pretrained_encoder=True):
    model = MambaPedOSA(config)
    if pretrained_encoder:
        model.encoder = load_sleepjepa_weights(model.encoder)
    return model
