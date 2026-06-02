import pytest
import numpy as np
import torch
import os
import h5py
import tempfile

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.augmentation import PedOSADataAugmentation
from data.dataset import PedOSADataset, get_dataloaders

def test_augmentation_range():
    aug = PedOSADataAugmentation()
    
    # Mock physiological ECG (e.g., normally around 0 with R-peaks up to 1 mV)
    ecg = np.zeros(7680)
    spo2 = np.ones(960) * 98.0
    
    # Force all augmentations to trigger
    aug.p_downsample = 1.0
    aug.p_emg = 1.0
    aug.p_baseline = 1.0
    aug.p_invert = 1.0
    aug.p_spo2_artifact = 1.0
    
    for _ in range(20):
        ecg_aug, spo2_aug = aug(ecg, spo2)
        # Check physiological bounds
        assert np.max(ecg_aug) <= 5.0
        assert np.min(ecg_aug) >= -5.0

def test_ptt_normalization():
    # If the distribution has mean 17.92, normalization should center it
    dummy_ptt_distribution = np.random.normal(17.92, 5.0, 1000)
    normalized = (dummy_ptt_distribution - 17.92) / 5.0
    assert np.isclose(np.mean(normalized), 0.0, atol=0.1)

@pytest.fixture
def mock_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy HDF5 files for train, val, nch
        splits_to_mock = [("train", "train_sub001"), ("val", "val_sub001"), ("nch", "nch_sub001")]
        
        for split, prefix in splits_to_mock:
            path = os.path.join(tmpdir, f"{prefix}.h5")
            with h5py.File(path, 'w') as f:
                # 60 seconds of data to yield at least a few 30s overlapping windows
                f.create_dataset('ecg', data=np.random.randn(256 * 60)) 
                f.create_dataset('spo2', data=np.ones(32 * 60) * 98.0)
                f.create_dataset('ptt_swing', data=np.random.randn(32 * 60) + 17.92)
                f.attrs['age_norm'] = 0.5
                f.attrs['bmi_z'] = 1.2
                f.attrs['sex'] = 1.0
                f.attrs['ahi_label'] = 5.0
                f.attrs['severity_class'] = 1
                f.attrs['is_longitudinal'] = False
                
        yield tmpdir

def test_nch_absence_in_train(mock_data_dir):
    train_dataset = PedOSADataset(mock_data_dir, split="train", augment=False)
    assert len(train_dataset) > 0
    for sample in train_dataset.samples:
        assert "nch_" not in os.path.basename(sample['file_path'])

def test_batch_shapes(mock_data_dir):
    train_loader, _, _, _ = get_dataloaders(mock_data_dir)
    batch = next(iter(train_loader))
    
    assert len(batch['ecg'].shape) == 2
    assert batch['ecg'].shape[1] == 7680
    
    assert len(batch['spo2'].shape) == 2
    assert batch['spo2'].shape[1] == 960
    
    assert len(batch['clinical'].shape) == 2
    assert batch['clinical'].shape[1] == 3
    
    assert len(batch['ptt_swing'].shape) == 1
    assert len(batch['severity_class'].shape) == 1
