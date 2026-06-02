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
        # We assume files are named like: {split}_subjectXXX.h5
        # e.g., train_sub001.h5, nch_sub001.h5
        search_pattern = os.path.join(self.data_dir, f"{self.split}_*.h5")
        file_paths = glob.glob(search_pattern)
        
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as f:
                ecg_len = len(f['ecg'])
                
                # Calculate number of valid overlapping windows
                num_windows = (ecg_len - self.ecg_window) // self.ecg_stride + 1
                
                # Extract subject-level metadata
                # Assuming these are stored as attributes or scalar datasets
                clinical = (
                    float(f.attrs.get('age_norm', 0.0)),
                    float(f.attrs.get('bmi_z', 0.0)),
                    float(f.attrs.get('sex', 0.0))
                )
                ahi_label = float(f.attrs.get('ahi_label', 0.0))
                severity_class = int(f.attrs.get('severity_class', 0))
                is_longitudinal = bool(f.attrs.get('is_longitudinal', False))
                
                for i in range(num_windows):
                    ecg_start = i * self.ecg_stride
                    spo2_start = i * self.spo2_stride
                    
                    self.samples.append({
                        'file_path': file_path,
                        'ecg_start': ecg_start,
                        'spo2_start': spo2_start,
                        'clinical': clinical,
                        'ahi_label': ahi_label,
                        'severity_class': severity_class,
                        'is_longitudinal': is_longitudinal
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        with h5py.File(sample_info['file_path'], 'r') as f:
            ecg_start = sample_info['ecg_start']
            spo2_start = sample_info['spo2_start']
            
            ecg = f['ecg'][ecg_start : ecg_start + self.ecg_window]
            spo2 = f['spo2'][spo2_start : spo2_start + self.spo2_window]
            
            # For PTT swing, assuming it's aligned with SpO2 or ECG.
            # If it's a dataset, we extract the window and take the mean to get a scalar.
            # Or if it's already an array of length num_windows, we could just index it.
            # Let's assume it's continuous at SpO2 frequency.
            if 'ptt_swing' in f:
                ptt_window = f['ptt_swing'][spo2_start : spo2_start + self.spo2_window]
                ptt_swing_scalar = float(np.mean(ptt_window))
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
