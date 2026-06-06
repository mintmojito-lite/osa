import os
import argparse
import numpy as np
import h5py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, matthews_corrcoef, brier_score_loss, average_precision_score
from pathlib import Path
import json

def extract_features_for_rf(data_dir, split_subjects):
    features = []
    labels = []
    severity = []
    
    for subj in split_subjects:
        multi_path = os.path.join(data_dir, f"{subj}_multimodal.h5")
        if not os.path.exists(multi_path):
            continue
            
        with h5py.File(multi_path, 'r') as f:
            ecg = f['ecg_epochs'][:]
            # Simple HRV mock features
            mean_ecg = np.mean(ecg, axis=1).mean()
            std_ecg = np.std(ecg, axis=1).mean()
            
            if 'ptt_swings' in f:
                mean_ptt = np.mean(f['ptt_swings'][:])
            else:
                mean_ptt = 0.0
                
            features.append([mean_ecg, std_ecg, mean_ptt])
            labels.append(f.attrs.get('computed_ahi', 0.0))
            
            sev = f.attrs.get('ahi_label', 'Normal')
            sev_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
            severity.append(sev_map.get(sev, 0))
            
    return np.array(features), np.array(labels), np.array(severity)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/features')
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    splits_path = base_dir / 'configs' / 'splits.json'
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    print("Extracting features for Random Forest...")
    X_train, y_train, s_train = extract_features_for_rf(args.data_dir, splits['train'])
    X_val, y_val, s_val = extract_features_for_rf(args.data_dir, splits['val'])
    
    if len(X_train) == 0:
        print("No training data found for baseline.")
        return
        
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    
    print(f"Random Forest Baseline MAE: {mae:.4f}")
    
if __name__ == '__main__':
    main()
