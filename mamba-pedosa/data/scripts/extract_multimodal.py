import os
import mne
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
import neurokit2 as nk
from tqdm import tqdm

def get_severity_label(ahi):
    if ahi < 1: return 'Normal'
    elif ahi < 5: return 'Mild'
    elif ahi < 10: return 'Moderate'
    else: return 'Severe'

def compute_ptt_swing(ecg_data, spo2_data, fs, epoch_len=30):
    # Detect R-peaks
    try:
        _, rpeaks = nk.ecg_peaks(ecg_data, sampling_rate=fs)
        r_peaks_idx = rpeaks['ECG_R_Peaks']
    except Exception:
        # Fallback if neurokit fails
        r_peaks_idx, _ = find_peaks(ecg_data, distance=fs*0.5)

    # Detect SpO2 peaks
    try:
        # Assuming SpO2/PPG signal resembles PPG
        spo2_peaks_idx, _ = find_peaks(spo2_data, distance=fs*0.5)
    except Exception:
        spo2_peaks_idx = np.array([])

    epoch_samples = int(epoch_len * fs)
    num_epochs = len(ecg_data) // epoch_samples
    
    ptt_swings = np.zeros(num_epochs)
    
    for i in range(num_epochs):
        start_idx = i * epoch_samples
        end_idx = start_idx + epoch_samples
        
        # Get peaks in this epoch
        epoch_r_peaks = r_peaks_idx[(r_peaks_idx >= start_idx) & (r_peaks_idx < end_idx)]
        epoch_s_peaks = spo2_peaks_idx[(spo2_peaks_idx >= start_idx) & (spo2_peaks_idx < end_idx)]
        
        ptts = []
        for r_peak in epoch_r_peaks:
            # Find the first SpO2 peak after the R-peak
            valid_s_peaks = epoch_s_peaks[epoch_s_peaks > r_peak]
            if len(valid_s_peaks) > 0:
                s_peak = valid_s_peaks[0]
                # PTT in ms
                ptt = (s_peak - r_peak) / fs * 1000.0
                if 100 < ptt < 500: # physiological bounds for PTT
                    ptts.append(ptt)
                    
        if len(ptts) > 1:
            ptt_swings[i] = np.max(ptts) - np.min(ptts)
        else:
            ptt_swings[i] = 0.0 # Missing or invalid PTT
            
    return ptt_swings

def process_multimodal(edf_path, h5_ecg_path, output_h5_path, manifest_row):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        fs = raw.info['sfreq']
        
        # Get SpO2 channel
        spo2_ch = None
        for ch in raw.info['ch_names']:
            if 'SPO2' in ch.upper() or 'PLETH' in ch.upper():
                spo2_ch = ch
                break
                
        if spo2_ch is None:
            print(f"Warning: No SpO2 channel found for {edf_path}")
            return False
            
        spo2_data = raw.get_data(picks=[spo2_ch])[0]
        
        # Load preprocessed ECG
        with h5py.File(h5_ecg_path, 'r') as f:
            ecg_data = f['ecg'][:]
            ecg_fs = f.attrs['fs']
            
        # We need both signals at same fs for alignment
        # Assuming SpO2 and ECG are roughly aligned from EDF
        # In a real pipeline, we'd resample SpO2 to match ecg_fs (256Hz)
        if fs != ecg_fs:
            from scipy.signal import resample_poly
            up = int(ecg_fs)
            down = int(fs)
            g = np.gcd(up, down)
            spo2_data = resample_poly(spo2_data, up // g, down // g)
            fs = ecg_fs

        # Truncate to min length
        min_len = min(len(ecg_data), len(spo2_data))
        ecg_data = ecg_data[:min_len]
        spo2_data = spo2_data[:min_len]
        
        ptt_swings = compute_ptt_swing(ecg_data, spo2_data, fs)
        
        # Extract annotations/events
        # MNE automatically parses EDF+ annotations into raw.annotations
        annotations = raw.annotations
        sleep_stages = []
        apnea_hypopnea_count = 0
        total_sleep_epochs = 0
        
        # Simplify annotation parsing
        # (Real EDFs have complex scoring rules. Here we count events containing 'apnea' or 'hypopnea')
        for ann in annotations:
            desc = ann['description'].lower()
            if 'apnea' in desc or 'hypopnea' in desc:
                apnea_hypopnea_count += 1
            if 'sleep' in desc or 'stage' in desc or 'n1' in desc or 'n2' in desc or 'n3' in desc or 'rem' in desc:
                total_sleep_epochs += ann['duration'] / 30.0 # roughly

        # If annotations don't provide AHI, fall back to manifest
        total_sleep_hours = total_sleep_epochs * 30.0 / 3600.0 if total_sleep_epochs > 0 else 0
        if total_sleep_hours > 0:
            computed_ahi = apnea_hypopnea_count / total_sleep_hours
        else:
            computed_ahi = manifest_row['ahi']
            
        severity = get_severity_label(computed_ahi)
        
        epoch_samples = int(30 * fs)
        num_epochs = len(ecg_data) // epoch_samples
        
        age = manifest_row['age']
        bmi_zscore = 0.0 # dummy value since manifest lacks bmi
        
        # Save features per epoch
        with h5py.File(output_h5_path, 'w') as f:
            ecg_epochs = ecg_data[:num_epochs*epoch_samples].reshape(num_epochs, epoch_samples)
            spo2_epochs = spo2_data[:num_epochs*epoch_samples].reshape(num_epochs, epoch_samples)
            
            f.create_dataset('ecg_epochs', data=ecg_epochs, compression='gzip')
            f.create_dataset('spo2_epochs', data=spo2_epochs, compression='gzip')
            f.create_dataset('ptt_swings', data=ptt_swings)
            
            f.attrs['age'] = age
            f.attrs['bmi_zscore'] = bmi_zscore
            f.attrs['ahi_label'] = severity
            f.attrs['computed_ahi'] = computed_ahi
            
        return True

    except Exception as e:
        print(f"Error processing multimodal {edf_path}: {e}")
        return False

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_dir / 'data' / 'processed' / 'manifest.csv'
    features_dir = base_dir / 'data' / 'features'
    
    if not manifest_path.exists():
        print("Manifest not found.")
        return
        
    df = pd.read_csv(manifest_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Multimodal"):
        edf_path = Path(row['file_path'])
        subject_id = row['subject_id']
        h5_ecg_path = features_dir / f"{subject_id}_ecg.h5"
        output_h5_path = features_dir / f"{subject_id}_multimodal.h5"
        
        if edf_path.exists() and h5_ecg_path.exists():
            process_multimodal(edf_path, h5_ecg_path, output_h5_path, row)
        else:
            print(f"Skipping {subject_id}: missing EDF or preprocessed ECG.")

if __name__ == '__main__':
    main()
