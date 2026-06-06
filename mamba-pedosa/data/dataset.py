import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentation import PedOSADataAugmentation
from .sampler import get_balanced_sampler_and_weights

class PedOSADataset(Dataset):
    def __init__(self, data_dir, split="train", augment=False):
        """
        Args:
            data_dir (str): Path to data/processed containing HDF5 files.
            split (str): "train", "val", "test", or "nch".
            augment (bool): Whether to apply training augmentations.
        """
        self.data_dir = data_dir
        self.split = split
        self.augment = augment
        self.transform = PedOSADataAugmentation() if augment else None
        
        self.window_size_sec = 30
        self.stride_sec = 15
        self.ecg_fs = 256
        self.spo2_fs = 32
        
        self.ecg_window = self.window_size_sec * self.ecg_fs     # 7680
        self.spo2_window = self.window_size_sec * self.spo2_fs   # 960
        self.ecg_stride = self.stride_sec * self.ecg_fs
        self.spo2_stride = self.stride_sec * self.spo2_fs
        
        self.samples = []
        self._load_metadata()

    def _load_metadata(self):
        # Read configs/splits.json
        import json
        
        # We assume base_dir is parent of data_dir
        base_dir = os.path.dirname(os.path.dirname(self.data_dir))
        splits_path = os.path.join(base_dir, 'configs', 'splits.json')
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
        # Map split names
        split_key = self.split
        if self.split == 'nch':
            split_key = 'test_nch'
            
        subject_ids = splits.get(split_key, [])
        
        for subject_id in subject_ids:
            # Note: the dataset uses ecg.h5 and multimodal.h5, but we can read from multimodal.h5
            # since it probably has all data or we read from both.
            # The dummy pipeline creates ecg.h5 and multimodal.h5, but multimodal.h5 has everything:
            # 'ecg_epochs', 'spo2_epochs', 'ptt_swings'.
            file_prefix = os.path.join(self.data_dir, f"{subject_id}")
            
            multi_file_path = f"{file_prefix}_multimodal.h5"
            if not os.path.exists(multi_file_path):
                continue
                
            with h5py.File(multi_file_path, 'r') as f:
                ecg_epochs = f['ecg_epochs']
                num_epochs = ecg_epochs.shape[0]
                
                # Extract subject-level metadata
                clinical = (
                    float(f.attrs.get('age', 0.0)),
                    float(f.attrs.get('bmi_zscore', 0.0)),
                    0.0 # Dummy sex
                )
                ahi_label = float(f.attrs.get('computed_ahi', 0.0))
                severity_class = f.attrs.get('ahi_label', 'Normal')
                severity_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
                severity_class_idx = severity_map.get(severity_class, 0)
                is_longitudinal = False
                
                for i in range(num_epochs):
                    self.samples.append({
                        'file_prefix': file_prefix,
                        'epoch_idx': i,
                        'clinical': clinical,
                        'ahi_label': ahi_label,
                        'severity_class': severity_class_idx,
                        'is_longitudinal': is_longitudinal
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_prefix = sample_info['file_prefix']
        
        multi_file = f"{file_prefix}_multimodal.h5"
        
        with h5py.File(multi_file, 'r') as f_multi:
            epoch_idx = sample_info['epoch_idx']
            
            ecg = f_multi['ecg_epochs'][epoch_idx]
            spo2 = f_multi['spo2_epochs'][epoch_idx]
            
            if 'ptt_swings' in f_multi:
                ptt_swing_scalar = float(f_multi['ptt_swings'][epoch_idx])
            else:
                ptt_swing_scalar = 17.92 # fallback default
                
        # Apply augmentations (only during training)
        if self.augment and self.transform is not None:
            ecg, spo2 = self.transform(ecg, spo2)
            
        # Normalize ECG (zero-mean, unit-variance per segment)
        ecg_mean = np.mean(ecg)
        ecg_std = np.std(ecg) + 1e-8
        ecg = (ecg - ecg_mean) / ecg_std
        
        # Normalize PTT swing
        ptt_swing_norm = (ptt_swing_scalar - 17.92) / 5.0
        
        # Convert to tensors
        ecg_tensor = torch.tensor(ecg, dtype=torch.float32)
        spo2_tensor = torch.tensor(spo2, dtype=torch.float32)
        clinical_tensor = torch.tensor(sample_info['clinical'], dtype=torch.float32)
        
        return {
            'ecg': ecg_tensor,
            'spo2': spo2_tensor,
            'ptt_swing': torch.tensor(ptt_swing_norm, dtype=torch.float32),
            'clinical': clinical_tensor,
            'ahi_label': torch.tensor(sample_info['ahi_label'], dtype=torch.float32),
            'severity_class': torch.tensor(sample_info['severity_class'], dtype=torch.long)
        }

def get_dataloaders(data_dir):
    """
    Returns train, val, and nch dataloaders.
    """
    # 1. Training DataLoader
    train_dataset = PedOSADataset(data_dir, split="train", augment=True)
    
    # Extract labels and longitudinal flags for sampler
    severity_classes = [s['severity_class'] for s in train_dataset.samples]
    is_longitudinal = [s['is_longitudinal'] for s in train_dataset.samples]
    
    if len(train_dataset) > 0:
        sampler, alpha = get_balanced_sampler_and_weights(severity_classes, is_longitudinal)
    else:
        sampler = None
        alpha = None
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        sampler=sampler
    )
    
    # 2. Validation DataLoader
    val_dataset = PedOSADataset(data_dir, split="val", augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 3. NCH External DataLoader
    nch_dataset = PedOSADataset(data_dir, split="nch", augment=False)
    nch_loader = DataLoader(
        nch_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, nch_loader, alpha
