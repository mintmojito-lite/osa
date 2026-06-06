import torch
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef
import torchmetrics

def compute_auprc(probs, targets):
    """
    Computes AUPRC (Average Precision Score) per class and macro average.
    probs: (N, C) predicted probabilities
    targets: (N,) integer class labels
    """
    num_classes = probs.shape[1]
    auprc_scores = {}
    
    # Binarize targets
    targets_onehot = np.eye(num_classes)[targets]
    
    # Per-class AUPRC
    for c in range(num_classes):
        # Only compute if there are positive samples in this class
        if np.sum(targets_onehot[:, c]) > 0:
            auprc = average_precision_score(targets_onehot[:, c], probs[:, c])
            auprc_scores[f'auprc_class_{c}'] = auprc
            
    # Macro AUPRC
    if auprc_scores:
        auprc_scores['auprc_macro'] = np.mean(list(auprc_scores.values()))
        
    return auprc_scores

def compute_brier_scores(probs, targets):
    """
    Computes Brier Score per class.
    probs: (N, C) predicted probabilities
    targets: (N,) integer class labels
    """
    num_classes = probs.shape[1]
    brier_scores = {}
    
    targets_onehot = np.eye(num_classes)[targets]
    
    for c in range(num_classes):
        bs = brier_score_loss(targets_onehot[:, c], probs[:, c])
        brier_scores[f'brier_class_{c}'] = bs
        
    brier_scores['brier_macro'] = np.mean(list(brier_scores.values()))
    return brier_scores

def compute_ece(probs, targets, n_bins=10):
    """
    Expected Calibration Error.
    probs: (N, C)
    targets: (N,)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        # Include 0 in the first bin
        if i == 0:
            in_bin = in_bin | (confidences == 0)
            
        prob_in_bin = np.mean(in_bin)
        if prob_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prob_in_bin
            
    return ece

def compute_mcc(preds, targets):
    """
    Matthews Correlation Coefficient.
    preds: (N,) integer class predictions
    targets: (N,) integer class labels
    """
    return matthews_corrcoef(targets, preds)
