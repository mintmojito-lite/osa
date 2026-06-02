import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler_and_weights(severity_classes, is_longitudinal):
    """
    Computes a WeightedRandomSampler for class balancing and returns focal loss weights.
    
    Args:
        severity_classes: Array-like of ints (0=Normal, 1=Mild, 2=Mod, 3=Severe)
        is_longitudinal: Array-like of bools indicating longitudinal sub-cohort
        
    Returns:
        sampler: torch.utils.data.WeightedRandomSampler
        alpha: torch.Tensor of focal loss weights [0.25, 0.5, 1.0, 2.0]
    """
    severity_classes = np.array(severity_classes, dtype=int)
    is_longitudinal = np.array(is_longitudinal, dtype=bool)
    
    num_samples = len(severity_classes)
    
    # Compute per-class counts
    # Assumes classes are 0, 1, 2, 3
    class_counts = np.bincount(severity_classes, minlength=4)
    
    # Weight = 1 / class_frequency
    # Avoid div by zero
    safe_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / safe_counts
    
    sample_weights = np.zeros(num_samples, dtype=float)
    
    for i in range(num_samples):
        cls = severity_classes[i]
        weight = class_weights[cls]
        
        # Oversample longitudinal sub-cohort by 3x
        if is_longitudinal[i]:
            weight *= 3.0
            
        sample_weights[i] = weight
        
    # Convert to torch tensor
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=num_samples,
        replacement=True
    )
    
    # Focal loss weight vector for [Normal, Mild, Mod, Severe]
    alpha = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32)
    
    return sampler, alpha
