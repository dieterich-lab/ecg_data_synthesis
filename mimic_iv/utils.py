import numpy as np


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
