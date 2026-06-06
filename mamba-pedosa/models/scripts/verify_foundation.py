import os
import sys
import torch
import logging
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.foundation import SleepJEPAEncoder, load_sleepjepa_weights, freeze_encoder
from models.fusion import MultiModalFusion

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_foundation():
    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/model_config.yaml'))
    cfg = OmegaConf.load(config_path)
    
    # 1. Initialize models
    encoder = SleepJEPAEncoder(
        channels=cfg.encoder_channels,
        strides=cfg.encoder_strides,
        dropout=cfg.dropout
    )
    
    fusion = MultiModalFusion(ssm_d_model=cfg.ssm_d_model)
    
    # 2. Try loading weights
    encoder = load_sleepjepa_weights(encoder)
    
    # 3. Freeze encoder blocks
    freeze_encoder(encoder, freeze_layers=[0, 1])
    
    # 4. Generate random batch
    B = 2
    ecg = torch.randn(B, 1, 7680)
    spo2 = torch.randn(B, 1, 960)
    ptt = torch.randn(B, 1)
    clinical = torch.randn(B, 3)
    
    # 5. Forward pass
    logging.info("Running forward pass...")
    ecg_features = encoder(ecg)
    
    assert ecg_features.shape == (B, 256, 960), f"Encoder output shape mismatch: {ecg_features.shape}"
    
    fused_features = fusion(ecg_features, spo2, ptt, clinical)
    assert fused_features.shape == (B, cfg.ssm_d_model, 960), f"Fusion output shape mismatch: {fused_features.shape}"
    
    logging.info(f"Forward pass successful. Output shape: {fused_features.shape}")
    
    # 6. Parameter count logic
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen

    enc_t, enc_train, enc_froz = count_parameters(encoder)
    fus_t, fus_train, fus_froz = count_parameters(fusion)
    
    total_params = enc_t + fus_t
    trainable_params = enc_train + fus_train
    frozen_params = enc_froz + fus_froz
    
    logging.info(f"Parameter Counts:")
    logging.info(f"  Encoder -> Total: {enc_t:,} | Trainable: {enc_train:,} | Frozen: {enc_froz:,}")
    logging.info(f"  Fusion  -> Total: {fus_t:,} | Trainable: {fus_train:,} | Frozen: {fus_froz:,}")
    logging.info(f"  Overall -> Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")
    
    # 7. Save Checkpoint
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/checkpoints'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'init_jepa.pt')
    
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'fusion_state_dict': fusion.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved initialized model checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    verify_foundation()
