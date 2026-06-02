import os
import sys
import glob
import h5py
import json
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.dataset import PedOSADataset
from data.sampler import get_balanced_sampler_and_weights

def compute_stats():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs'))
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    stats = {
        'total_subjects': 0,
        'splits': {},
        'ptt_swing': {},
        'class_imbalance': {}
    }
    
    all_ptt_swings = []
    
    for split in ['train', 'val', 'test', 'nch']:
        dataset = PedOSADataset(data_dir, split=split, augment=False)
        
        files = glob.glob(os.path.join(data_dir, f"{split}_*.h5"))
        num_subjects = len(files)
        stats['total_subjects'] += num_subjects
        
        ages = []
        severity_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for f in files:
            with h5py.File(f, 'r') as h5f:
                age = float(h5f.attrs.get('age_norm', 0.0))
                ages.append(age)
                
                if 'ptt_swing' in h5f:
                    ptt = h5f['ptt_swing'][:]
                    all_ptt_swings.extend(ptt)
                
        for sample in dataset.samples:
            cls = sample['severity_class']
            severity_counts[cls] += 1
            
        if len(ages) > 0:
            age_dist = {
                'mean': float(np.mean(ages)),
                'std': float(np.std(ages)),
                'min': float(np.min(ages)),
                'max': float(np.max(ages))
            }
        else:
            age_dist = None
            
        stats['splits'][split] = {
            'subjects': num_subjects,
            'epochs_per_class': severity_counts,
            'age_distribution': age_dist
        }

    # PTT swing stats
    if len(all_ptt_swings) > 0:
        ptt_arr = np.array(all_ptt_swings)
        mean_ptt = float(np.mean(ptt_arr))
        std_ptt = float(np.std(ptt_arr))
        pct_below = float(np.mean(ptt_arr < 17.92) * 100)
    else:
        mean_ptt = 17.92
        std_ptt = 5.0
        pct_below = 50.0

    stats['ptt_swing'] = {
        'mean': mean_ptt,
        'std': std_ptt,
        'percent_below_17_92': pct_below
    }
    
    # Class imbalance
    train_dataset = PedOSADataset(data_dir, split="train", augment=False)
    if len(train_dataset) > 0:
        severity_classes = [s['severity_class'] for s in train_dataset.samples]
        is_longitudinal = [s['is_longitudinal'] for s in train_dataset.samples]
        
        counts_before = np.bincount(severity_classes, minlength=4)
        ratio_before = counts_before / counts_before.sum()
        
        sampler, _ = get_balanced_sampler_and_weights(severity_classes, is_longitudinal)
        weights = sampler.weights.numpy()
        
        probs = weights / weights.sum()
        expected_counts_after = np.zeros(4)
        for i, cls in enumerate(severity_classes):
            expected_counts_after[cls] += probs[i]
            
        stats['class_imbalance'] = {
            'before_sampling_ratio': ratio_before.tolist(),
            'after_sampling_expected_ratio': expected_counts_after.tolist()
        }
    
    out_path = os.path.join(config_dir, 'dataset_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Stats computed and saved to {out_path}")

if __name__ == "__main__":
    compute_stats()
