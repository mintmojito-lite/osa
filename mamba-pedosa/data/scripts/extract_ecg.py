import os
import mne
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly, butter, filtfilt
from tqdm import tqdm

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_filter(data, b, a):
    return filtfilt(b, a, data)

def process_ecg(edf_path, output_h5_path, target_fs=256):
    try:
        # Load EDF without preloading into memory initially
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        
        # Identify ECG channel
        ch_names = raw.info['ch_names']
        ecg_ch = None
        for ch in ch_names:
            if 'ECG' in ch.upper() or 'EKG' in ch.upper() or 'LEAD II' in ch.upper():
                ecg_ch = ch
                break
                
        if ecg_ch is None:
            print(f"Warning: No ECG channel found in {edf_path}")
            return False
            
        # Extract ECG data
        raw.pick_channels([ecg_ch])
        raw.load_data()
        data = raw.get_data()[0]
        orig_fs = raw.info['sfreq']
        
        # Resample to 256 Hz
        if orig_fs != target_fs:
            # We use resample_poly, which needs integer factors
            # Usually orig_fs is an integer like 200, 256, 500, 512
            up = int(target_fs)
            down = int(orig_fs)
            # Find greatest common divisor to minimize up/down factors
            g = np.gcd(up, down)
            data = resample_poly(data, up // g, down // g)
            
        # Bandpass filter 0.5 - 40 Hz
        b_bp, a_bp = butter_bandpass(0.5, 40.0, target_fs, order=4)
        data = apply_filter(data, b_bp, a_bp)
        
        # Highpass filter 0.5 Hz (Baseline wander removal)
        b_hp, a_hp = butter_highpass(0.5, target_fs, order=4)
        data = apply_filter(data, b_hp, a_hp)
        
        # Convert to float32
        data = data.astype(np.float32)
        
        # Save to HDF5
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('ecg', data=data, compression='gzip')
            f.attrs['fs'] = target_fs
            f.attrs['original_fs'] = orig_fs
            f.attrs['source_file'] = str(edf_path)
            
        return True
    except Exception as e:
        print(f"Error processing {edf_path}: {e}")
        return False

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_dir / 'data' / 'processed' / 'manifest.csv'
    features_dir = base_dir / 'data' / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run download_data.py first.")
        return
        
    df = pd.read_csv(manifest_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECG"):
        edf_path = Path(row['file_path'])
        subject_id = row['subject_id']
        output_h5_path = features_dir / f"{subject_id}_ecg.h5"
        
        if edf_path.exists():
            process_ecg(edf_path, output_h5_path)
        else:
            print(f"File not found: {edf_path}")

if __name__ == '__main__':
    main()
