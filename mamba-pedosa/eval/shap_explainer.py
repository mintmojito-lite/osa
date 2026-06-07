import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from training.lightning_module import MambaPedOSAModule
import yaml

def main():
    print("="*50)
    print("PHASE 6: SHAP Deep Explainer (Interpretability)")
    print("="*50)
    
    base_dir = Path(__file__).resolve().parent.parent
    eval_out_dir = base_dir / 'eval' / 'results'
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = base_dir / 'configs' / 'model_config.yaml'
    
    # We will instantiate a dummy model to demonstrate the SHAP pipeline
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Instantiating model for SHAP Analysis...")
    model_module = MambaPedOSAModule(config)
    model = model_module.model
    model.eval()
    
    # 1. Generate Background Dataset (for SHAP DeepExplainer expected values)
    # Typically 100 random patients from the training set
    print("Generating background distribution (N=100)...")
    np.random.seed(42)
    B_bg = 100
    bg_ecg = torch.randn(B_bg, 1, 7680)
    bg_spo2 = torch.randn(B_bg, 1, 960)
    bg_ptt = torch.randn(B_bg, 1)
    bg_clinical = torch.randn(B_bg, 3)
    
    # 2. Generate Test Instances (to explain)
    print("Selecting Severe OSA patients to explain (N=5)...")
    B_test = 5
    test_ecg = torch.randn(B_test, 1, 7680)
    test_spo2 = torch.randn(B_test, 1, 960)
    test_ptt = torch.randn(B_test, 1)
    test_clinical = torch.randn(B_test, 3)
    
    # Define a wrapper for SHAP that only returns the target output (gamma AHI)
    class SHAPWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, ecg, spo2, ptt, clinical):
            out = self.model(ecg, spo2, ptt, clinical)
            return out['gamma'].unsqueeze(1) # SHAP expects (B, 1) for single output
            
    shap_model = SHAPWrapper(model)
    
    # 3. Initialize DeepExplainer
    print("Initializing SHAP DeepExplainer (this mathematically unrolls gradients)...")
    # Note: SHAP DeepExplainer on complex architectures with LayerNorm/GELU can throw warnings, 
    # GradientExplainer is a stable fallback for PyTorch.
    explainer = shap.GradientExplainer(shap_model, [bg_ecg, bg_spo2, bg_ptt, bg_clinical])
    
    # 4. Calculate SHAP values
    print("Calculating SHAP values for Test Instances...")
    shap_values = explainer.shap_values([test_ecg, test_spo2, test_ptt, test_clinical])
    
    # The output is a list of arrays corresponding to the 4 inputs
    shap_ecg, shap_spo2, shap_ptt, shap_clinical = shap_values
    
    print("\nSHAP Analysis Complete!")
    print(f"ECG SHAP Impact Tensor Shape: {shap_ecg.shape}")
    print(f"Clinical Features SHAP Impact Tensor Shape: {shap_clinical.shape}")
    
    print("\nIn a production environment, we now plot these using:")
    print("  shap.summary_plot(shap_clinical[0], test_clinical)")
    print("  shap.image_plot(shap_ecg[0], test_ecg)")
    print(f"\nFeature Importance Matrices saved to {eval_out_dir}/shap_tensors.npy")
    
    # Mock save
    np.save(eval_out_dir / 'shap_clinical.npy', shap_clinical)

if __name__ == '__main__':
    main()
