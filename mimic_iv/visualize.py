import os

import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk

idx = 10
sample = 49
exp = 'filtered_352000'

real_ecg = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{idx}_real_ecg.npy')
gen_ecg = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{idx}_gen_ecg.npy')
labels = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{idx}_labels.npy')

print(real_ecg.shape, gen_ecg.shape, labels.shape)


def denormalize_data(normalized_data, min_val, max_val):
    """
    De-normalizes the data using the original min and max values.

    Args:
        normalized_data (numpy.ndarray): The normalized data to be de-normalized.
        min_val (numpy.ndarray): The minimum values used for normalization (lead-wise).
        max_val (numpy.ndarray): The maximum values used for normalization (lead-wise).

    Returns:
        numpy.ndarray: The de-normalized data in the original range.
    """
    denormalized_data = normalized_data * (max_val - min_val) + min_val
    return denormalized_data


# De-normalize the generated ECG data
train_min = np.load('processed_data/subset/mimic_train_min.npy')
train_max = np.load('processed_data/subset/mimic_train_max.npy')
generated_data_denormalized = denormalize_data(gen_ecg, train_min, train_max)

real_ecg_denormalized = denormalize_data(real_ecg, train_min, train_max)

time = np.arange(0, 1000) / 100  # Time in seconds

plt.figure(figsize=(12, 4))

# plot real denormalized ecgs
plt.plot(time, real_ecg_denormalized[sample, 0], label="Real Healthy ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Real ECG")
plt.xticks(np.arange(0, 11, 1))  # From 0 to 10 seconds, step of 1
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'sssd_label_cond/generated_ecgs_{exp}/real_ecg_{idx}_{sample}.png')

# plot generated denormalized ecgs
plt.clf()
plt.plot(time, generated_data_denormalized[sample, 0], color='orange', label="Generated Healthy ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Generated ECG")
plt.xticks(np.arange(0, 11, 1))  # From 0 to 10 seconds, step of 1
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'sssd_label_cond/generated_ecgs_{exp}/gen_ecg_{idx}_{sample}.png')
