import numpy as np
from scipy.signal import resample
import scipy.signal as signal


def downsample_ecg(ecg_data, target_points):
    """
    Downsamples the ECG data to the target number of points.

    ecg_data: numpy array of shape (num_leads, original_length)
    target_points: int, the number of points to resample to (e.g., 1000)

    Returns:
    downsampled_data: numpy array of shape (num_leads, target_points)
    """
    num_leads, original_length = ecg_data.shape
    downsampled_data = np.zeros((num_leads, target_points))

    for lead in range(num_leads):
        downsampled_data[lead, :] = resample(ecg_data[lead, :], target_points)

    return downsampled_data


def remove_nan(data_ptbxl, labels_ptbxl):
    # Filter data by removal of NaN values
    mask = ~np.isnan(data_ptbxl).any(axis=(1, 2))
    filtered_data = data_ptbxl[mask]
    filtered_labels = labels_ptbxl[mask]
    print("Original shape:", data_ptbxl.shape)
    print("Filtered shape:", filtered_data.shape)
    print("Filtered labels shape:", filtered_labels.shape)
    return filtered_data, filtered_labels


def detect_r_peaks(ecg_signal, sampling_rate=250):
    """Detects R-peaks in an ECG signal using peak detection."""
    b, a = signal.butter(3, [0.5, 49.5], btype='bandpass', fs=sampling_rate)
    filtered_ecg = signal.filtfilt(b, a, ecg_signal)
    peaks, _ = signal.find_peaks(filtered_ecg, distance=sampling_rate*0.6, height=np.mean(filtered_ecg) + np.std(filtered_ecg))
    return peaks

def compute_rr_intervals(r_peaks, sampling_rate=250):
    """Computes RR intervals (in milliseconds)."""
    if len(r_peaks) < 2:
        return np.array([])  # Not enough peaks to compute RR intervals
    rr_intervals = np.diff(r_peaks) * (1000 / sampling_rate)
    return rr_intervals

def detect_afib(rr_intervals):
    """Checks if an ECG is likely AFib based on HR variability."""
    if len(rr_intervals) < 2:
        return False  # Not enough data to analyze

    sdnn = np.std(rr_intervals)  # Standard Deviation of RR intervals
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # RMSSD

    # AFib Thresholds: (these can be adjusted)
    return sdnn > 50 and rmssd > 50  # High variability suggests AFib

def find_afib_ecgs(ecg_dataset, sampling_rate=250):
    """Finds indices of ECGs that are likely AFib."""
    afib_indices = []

    for i, ecg_signal in enumerate(ecg_dataset):
        ecg_signal = ecg_signal[1]
        r_peaks = detect_r_peaks(ecg_signal, sampling_rate)
        rr_intervals = compute_rr_intervals(r_peaks, sampling_rate)

        if detect_afib(rr_intervals):
            afib_indices.append(i)  # Save index of suspected AFib ECG

    return afib_indices
