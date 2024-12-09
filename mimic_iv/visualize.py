import matplotlib.pyplot as plt
import numpy as np

LEAD_SEQ = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_min_max_values():
    """
    Load minimum and maximum values used for normalization.

    Returns:
        tuple: Two numpy arrays representing minimum and maximum values.
    """
    train_min = np.load('processed_data/latest/mimic_train_min.npy')
    train_max = np.load('processed_data/latest/mimic_train_max.npy')
    return train_min, train_max


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
    return normalized_data * (max_val - min_val) + min_val


def plot_all_leads(data, time, title, output_path, color="blue"):
    """
    Plots all 12 leads of ECG data in a 6x2 grid.

    Args:
        data (numpy.ndarray): ECG data to plot, shape (samples, leads, time).
        time (numpy.ndarray): Time array corresponding to the x-axis.
        title (str): Title of the plot.
        output_path (str): Path to save the plot.
        color (str): Color of the ECG lines.
    """
    fig, axes = plt.subplots(6, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(12):
        row, col = divmod(i, 2)
        axes[row, col].plot(time, data[sample_idx, i], color=color)
        axes[row, col].set_title(f"Lead {LEAD_SEQ[i]}")
        axes[row, col].set_xlabel("Time (s)")
        axes[row, col].set_ylabel("Amplitude (mV)")
        axes[row, col].grid(True)

    plt.tight_layout()  # Prevent overlap with suptitle
    plt.savefig(output_path)
    plt.close(fig)


def plot_single_lead(data, time, title, output_path, lead=0, color="blue"):
    """
    Plots a single lead of ECG data.

    Args:
        data (numpy.ndarray): ECG data to plot, shape (samples, leads, time).
        time (numpy.ndarray): Time array corresponding to the x-axis.
        title (str): Title of the plot.
        output_path (str): Path to save the plot.
        lead (int): Index of the lead to plot.
        color (str): Color of the ECG line.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(time, data[sample_idx, lead], color=color, label=f"Lead I")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title(title)
    plt.xticks(np.arange(0, 11, 1))
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_ecgs(gen_ecg_denormalized, all_leads=False, lead=0):
    """
    Visualizes ECG data by plotting generated ECGs.

     Args:
        gen_ecg_denormalized (numpy.ndarray): De-normalized generated ECG data, shape (samples, leads, time).
        all_leads (bool): Whether to plot all leads or just a single lead.
    """
    time = np.arange(0, 1000) / 100  # Time in seconds

    if all_leads:
        # Plot all 12 leads for generated ECG
        plot_all_leads(gen_ecg_denormalized, time,
                       "Generated ECGs (All 12 Leads)",
                       f'sssd_label_cond/generated_ecgs_{exp}/generated_ecgs_batch_{batch_idx}_{sample_idx}.png',
                       color="orange")
    else:
        # Plot single lead for generated ECG
        plot_single_lead(gen_ecg_denormalized, time,
                         f"Generated ECG (Lead {LEAD_SEQ[lead]})",
                         f'sssd_label_cond/generated_ecgs_{exp}/gen_ecg_batch_{batch_idx}_{sample_idx}.png',
                         lead=lead,
                         color="orange")


if __name__ == "__main__":
    batch_idx = 2
    sample_idx = 9
    exp = 'filtered_latest'

    # Load data
    gen_ecg = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{batch_idx}_gen_ecg.npy')
    labels = np.load(f'sssd_label_cond/generated_ecgs_{exp}/{batch_idx}_labels.npy')

    print(gen_ecg.shape, labels.shape)

    # Load min and max values
    train_min, train_max = load_min_max_values()

    # de-normalize the ecgs
    gen_ecg_denormalized = denormalize_data(gen_ecg, train_min, train_max)

    # Visualize ECGs
    visualize_ecgs(gen_ecg, all_leads=False, lead=0)
