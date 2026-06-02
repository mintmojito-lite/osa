import numpy as np
from scipy import signal
import torch

class PedOSADataAugmentation:
    def __init__(self):
        # Probabilities
        self.p_downsample = 0.3
        self.p_emg = 0.4
        self.p_baseline = 0.5
        self.p_invert = 0.2
        self.p_spo2_artifact = 0.25
        
        self.original_fs = 256
        self.downsample_fs = 100

    def __call__(self, ecg, spo2):
        """
        Apply augmentations to ECG and SpO2 signals.
        ecg: (7680,) numpy array
        spo2: (960,) numpy array
        Returns augmented (ecg, spo2)
        """
        ecg_aug = ecg.copy()
        spo2_aug = spo2.copy()

        # 1. Stochastic downsampling (simulates consumer wearables)
        if np.random.rand() < self.p_downsample:
            # Downsample to 100Hz
            num_samples_down = int(len(ecg_aug) * self.downsample_fs / self.original_fs)
            ecg_down = signal.resample(ecg_aug, num_samples_down)
            # Upsample back to 256Hz
            ecg_aug = signal.resample(ecg_down, len(ecg_aug))

        # 2. EMG noise injection (pediatric motion artifacts)
        if np.random.rand() < self.p_emg:
            # Generate Gaussian noise
            noise = np.random.normal(0, 1, len(ecg_aug))
            
            # Bandpass filter (20-450Hz). Nyquist is 128Hz, so 450Hz is above Nyquist.
            # We can only filter up to Nyquist. For a 256Hz signal, max freq is 128Hz.
            # The prompt says 20-450Hz, but at 256Hz sampling rate, we can only do 20-128Hz.
            # Let's use 20-125Hz to be safe.
            nyq = 0.5 * self.original_fs
            low = 20 / nyq
            high = 125 / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_noise = signal.filtfilt(b, a, noise)
            
            # Calculate signal power and noise power
            signal_power = np.mean(ecg_aug ** 2)
            noise_power = np.mean(filtered_noise ** 2)
            
            # Random SNR between 10 and 20 dB
            snr_db = np.random.uniform(10, 20)
            
            # Calculate required noise multiplier to achieve desired SNR
            # SNR_dB = 10 * log10(P_signal / P_noise)
            # P_noise = P_signal / (10 ** (SNR_dB / 10))
            target_noise_power = signal_power / (10 ** (snr_db / 10))
            
            if noise_power > 0:
                multiplier = np.sqrt(target_noise_power / noise_power)
                ecg_aug += filtered_noise * multiplier

        # 3. Baseline wander (sinusoidal drift)
        if np.random.rand() < self.p_baseline:
            f = np.random.uniform(0.05, 0.5)
            amplitude = np.random.uniform(0.05, 0.2)
            t = np.arange(len(ecg_aug)) / self.original_fs
            phase = np.random.uniform(0, 2 * np.pi)
            drift = amplitude * np.sin(2 * np.pi * f * t + phase)
            ecg_aug += drift

        # 4. Lead polarity inversion
        if np.random.rand() < self.p_invert:
            ecg_aug = -ecg_aug

        # 5. SpO2 artifact (zero-out 5-15% of samples)
        if np.random.rand() < self.p_spo2_artifact:
            # Calculate how many samples to zero out
            percent_to_zero = np.random.uniform(0.05, 0.15)
            num_zero = int(len(spo2_aug) * percent_to_zero)
            
            # Create a contiguous block of zeros to simulate signal loss, or random dropout?
            # Usually artifact is a contiguous block or bursts. Let's do random scattered dropout to be safe.
            zero_indices = np.random.choice(len(spo2_aug), num_zero, replace=False)
            spo2_aug[zero_indices] = 0.0

        return ecg_aug, spo2_aug
