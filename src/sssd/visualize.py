import os

import pandas as pd
import sys

import numpy as np
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root_dir, 'src/ptb_xl/clinical_ts'))

real_samples = np.load('./src/sssd/ptbxl_test_data.npy')


# cond_labels = np.load('./sssd_label_cond/ch256_T200_betaT0.02/0_labels.npy')
# gen_samples = np.load('./sssd_label_cond/ch256_T200_betaT0.02/0_samples.npy')


def get_label_names():
    blind_test_real, blind_test_label, blind_test_synth, index_val = [], [], [], []
    for s in range(5):
        real_ecgs = real_samples[400 * s: (400 * s) + 400, :]
        cond_labels = np.load('./sssd_label_cond/ch256_T200_betaT0.02/{}_labels.npy'.format(s))
        gen_samples = np.load('./sssd_label_cond/ch256_T200_betaT0.02/{}_samples.npy'.format(s))
        for i, l in enumerate(cond_labels):
            indices = [idx for idx, val in enumerate(l) if val]
            if len(indices) == 1:
                # and (indices[0] in [34, 35, 36, 5, 4, 9, 11, 12, 45, 46, 43, 47, 17, 50, 18, 53]):
                index_val.append(indices[0])
                blind_test_real.append(real_ecgs[i])
                blind_test_label.append(l)
                blind_test_synth.append(gen_samples[i, :])
    np.save('blind_test_ecgs_real.npy', np.array(blind_test_real))
    np.save('blind_test_ecgs_labels.npy', np.array(blind_test_label))
    np.save('blind_test_ecgs_synth.npy', np.array(blind_test_synth))
    counts = {
        val: index_val.count(val)
        for val in sorted(set(index_val))
    }
    visualize_ecg(np.array(blind_test_real), np.array(blind_test_label), index_val)
    # visualize_ecg(np.array(blind_test_synth), np.array(blind_test_label), index_val)


def visualize_ecg(samples, labels, index_val):
    print(samples.shape, labels.shape, len(index_val))
    diag = pd.read_csv('src/ptb_xl/data_folder_ptb_xl/scp_statements.csv')
    diag_desc = diag['description']
    # Plot the 12-lead ECG with diagnosis labels
    diag_file = open(f'./generated_ecg_plots/synth/labels.csv', 'w')
    for si in range(samples.shape[0]):
        diagnosis = [diag_desc[i] for i, l in enumerate(labels[si]) if l]
        diag_file.write(str(diagnosis[0]) + '\n')
        fig, axs = plt.subplots(12, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.5)
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'v5', 'V6']
        t = np.linspace(0, 10, 1000)
        for lead, ax in zip(leads, axs.ravel()):
            # ax.set_ylabel('Amplitude (mV)')
            # ax.set_xlabel('Time (s)')
            idx = leads.index(lead)
            ax.plot(t, samples[si, idx], 'b', linewidth=.7)
            ax.set_title(f'Lead {lead.upper()}')
        # plt.suptitle(f'ECG Diagnosis - {diagnosis}')
        fig.text(0.5, 0.08, 'Time (s)', ha='center')
        fig.text(0.08, 0.5, 'Amplitude (mV)', va='center', rotation='vertical')
        plt.savefig(f'./generated_ecg_plots/synth/ecg{index_val[si]}b{si}{0}.png')
        plt.clf()
    diag_file.close()


get_label_names()
