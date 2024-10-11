import matplotlib.pyplot as plt
import numpy as np

from utils import normalize_data

idx = 2
sample = 9
exp = 'mimic_110000'

real_ecg = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{idx}_real_ecg.npy')
gen_ecg = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{idx}_gen_ecg.npy')
gen_ecg_normalized = gen_ecg  # normalize_data(gen_ecg)
labels = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{idx}_labels.npy')

# real_ecg = np.load('src/sssd/ptbxl_test_data.npy')
# real_ecg = real_ecg[0:50]
# gen_ecg = np.load(f'sssd_label_cond/ch256_T200_betaT0.02/{idx}_gen_samples.npy')
# gen_ecg_normalized = gen_ecg  # normalize_data(gen_ecg)
# labels = np.load(f'sssd_label_cond/ch256_T200_betaT0.02/{idx}_labels.npy')

print(real_ecg.shape, gen_ecg.shape, labels.shape)

# Plotting the ECGs
plt.figure(figsize=(14, 6))

# Plot real ECG
plt.subplot(2, 1, 1)
plt.plot(real_ecg[sample, 0], label=f'Real ECG (Label: {labels[sample, 0]})')
plt.title('Real ECG')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

# Plot generated ECG
plt.subplot(2, 1, 2)
plt.plot(gen_ecg_normalized[sample, 0], label=f'Generated ECG (Label: {labels[sample, 0]})', color='orange')
plt.title('Generated ECG')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'sssd_label_cond/generated_ecgs_{exp}/real_vs_gen_ecg_{idx}_{sample}.png')


