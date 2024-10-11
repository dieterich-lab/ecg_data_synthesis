import numpy as np
from scipy.signal import resample


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


def normalize_data(filtered_data):
    # Normalize dataset at each lead level
    normalized_data = np.zeros_like(filtered_data)
    for i in range(filtered_data.shape[1]):
        lead = filtered_data[:, i, :]
        min_val = np.min(lead, axis=-1, keepdims=True)
        max_val = np.max(lead, axis=-1, keepdims=True)
        # Avoid division by zero in case max_val equals min_val
        range_val = np.maximum(max_val - min_val, 1e-6)
        normalized_data[:, i, :] = (lead - min_val) / range_val
    return normalized_data
