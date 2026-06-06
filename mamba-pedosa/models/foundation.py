import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import logging

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, padding=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.gelu(self.bn(self.conv(x))))

class SleepJEPAEncoder(nn.Module):
    """
    1D-Conv encoder for raw ECG, compatible with SleepJEPA or randomly initialized.
    Takes (B, 1, 7680) and outputs (B, 256, 960).
    """
    def __init__(self, channels=[1, 64, 128, 256], strides=[2, 2, 2], dropout=0.1):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        in_ch = channels[0]
        
        for out_ch, stride in zip(channels[1:], strides):
            # kernel=7, padding=3 with stride=2 halves the sequence length
            self.blocks.append(ConvBlock(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, dropout=dropout))
            in_ch = out_ch
            
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def load_sleepjepa_weights(model: SleepJEPAEncoder, repo_id="sleepjepa/weights", filename="encoder.pt"):
    """
    Attempts to load pre-trained weights from Hugging Face Hub.
    Gracefully falls back to random initialization if not found.
    """
    try:
        logging.info(f"Attempting to download SleepJEPA weights from {repo_id}/{filename}...")
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logging.info("Successfully loaded SleepJEPA weights.")
    except Exception as e:
        logging.warning(f"Could not load pre-trained SleepJEPA weights: {e}. Falling back to random initialization.")
    return model

def freeze_encoder(model: SleepJEPAEncoder, freeze_layers=[0, 1]):
    """
    Freezes specified blocks in the encoder.
    """
    for idx, block in enumerate(model.blocks):
        if idx in freeze_layers:
            for param in block.parameters():
                param.requires_grad = False
        else:
            for param in block.parameters():
                param.requires_grad = True

def get_finetuning_param_groups(encoder: nn.Module, rest_of_model: nn.Module, base_lr: float, epoch: int, unfreeze_epoch: int = 11):
    """
    Returns parameter groups for the optimizer, implementing the fine-tuning schedule.
    Frozen for epochs 0-10 (encoder params not returned, or returned with 0 LR),
    unfrozen with LR/10 from epoch 11.
    """
    if epoch < unfreeze_epoch:
        # Encoder is frozen (requires_grad might be False, but we explicitly exclude or 0 LR)
        return [
            {'params': rest_of_model.parameters(), 'lr': base_lr}
        ]
    else:
        # Unfrozen: ensure all encoder parameters require grad
        for param in encoder.parameters():
            param.requires_grad = True
        return [
            {'params': rest_of_model.parameters(), 'lr': base_lr},
            {'params': encoder.parameters(), 'lr': base_lr / 10.0}
        ]
