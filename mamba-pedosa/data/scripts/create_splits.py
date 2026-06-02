import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_dir / 'data' / 'processed' / 'manifest.csv'
    splits_path = base_dir / 'configs' / 'splits.json'
    
    if not manifest_path.exists():
        print("Manifest not found.")
        return
        
    df = pd.read_csv(manifest_path)
    
    # Separate datasets
    bch_df = df[df['dataset'] == 'BCH'].copy()
    nch_df = df[df['dataset'] == 'NCH'].copy()
    
    # Extract just the subject IDs and labels for stratified splitting
    bch_subjects = bch_df['subject_id'].values
    bch_labels = bch_df['severity_label'].values
    
    # 80/10/10 split
    # First, split into train (80%) and temp (20%)
    # Some classes might be too small for stratified splitting if we have very few simulated samples,
    # so we add a fallback for small datasets.
    try:
        train_subj, temp_subj, train_labels, temp_labels = train_test_split(
            bch_subjects, bch_labels, test_size=0.2, stratify=bch_labels, random_state=42
        )
        # Split temp into val (50% of 20% = 10%) and test (50% of 20% = 10%)
        val_subj, test_subj = train_test_split(
            temp_subj, test_size=0.5, stratify=temp_labels, random_state=42
        )
    except ValueError as e:
        print("Warning: Stratified split failed (likely due to too few samples per class). Falling back to random split.")
        train_subj, temp_subj = train_test_split(bch_subjects, test_size=0.2, random_state=42)
        if len(temp_subj) > 1:
            val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42)
        else:
            val_subj, test_subj = temp_subj, []
        
    nch_subjects = nch_df['subject_id'].values.tolist()
    
    # Verify no overlap
    bch_set = set(bch_subjects)
    nch_set = set(nch_subjects)
    assert len(bch_set.intersection(nch_set)) == 0, "Data leakage detected: NCH subjects in BCH!"
    
    splits = {
        'train': list(train_subj),
        'val': list(val_subj),
        'test': list(test_subj),
        'test_nch': nch_subjects
    }
    
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Splits saved to {splits_path}")
    print(f"Train size: {len(splits['train'])}")
    print(f"Val size:   {len(splits['val'])}")
    print(f"Test size:  {len(splits['test'])}")
    print(f"NCH size:   {len(splits['test_nch'])}")

if __name__ == '__main__':
    main()
