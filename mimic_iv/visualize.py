import argparse

import matplotlib.pyplot as plt
import numpy as np

LEAD_SEQ = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def plot_all_leads(data, time, title, output_path, color="blue", sample_idx=0):
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


def plot_single_lead(data, time, title, output_path, lead=0, color="blue", sample_idx=0):
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


def visualize_ecgs(gen_ecg, batch_idx, sample_idx, all_leads=False, lead=0):
    """
    Visualizes ECG data by plotting generated ECGs.

     Args:
        gen_ecg (numpy.ndarray): Generated ECG data, shape (samples, leads, time).
        all_leads (bool): Whether to plot all leads or just a single lead.
    """
    time = np.arange(0, 1000) / 100  # Time in seconds

    if all_leads:
        # Plot all 12 leads for generated ECG
        plot_all_leads(gen_ecg, time,
                       "Generated ECGs (All 12 Leads)",
                       f'output/generated_ecgs/gen_ecgs_all_leads_{batch_idx}_{sample_idx}.png',
                       color="orange",
                       sample_idx=sample_idx
                       )
    else:
        # Plot single lead for generated ECG
        plot_single_lead(gen_ecg, time,
                         f"Generated ECG (Lead {LEAD_SEQ[lead]})",
                         f'output/generated_ecgs/gen_ecg_lead0_{batch_idx}_{sample_idx}.png',
                         lead=lead,
                         color="orange",
                         sample_idx=sample_idx),

    print('Plotted ECGs Successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Plotting Script")
    parser.add_argument("--batch_idx", type=int, required=True, help="Batch index")
    parser.add_argument("--sample_idx", type=int, required=True, help="Sample index")
    parser.add_argument("--all_leads", action="store_true", help="Plot all leads")

    args = parser.parse_args()

    # Load data
    gen_ecg = np.load(f'output/generated_ecgs/{args.batch_idx}_gen_ecg.npy')
    labels = np.load(f'output/generated_ecgs/{args.batch_idx}_labels.npy')

    # Visualize ECG
    visualize_ecgs(gen_ecg, args.batch_idx, args.sample_idx, all_leads=args.all_leads)
