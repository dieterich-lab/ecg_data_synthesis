import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import wfdb
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import downsample_ecg
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=sys.maxsize)


def get_label_categories(row, df):
    if row['Diagnosis'] in df['Diagnosis\n(Single and Multi-labels)'].values:
        row['Category'] = df.loc[
            df['Diagnosis\n(Single and Multi-labels)'] == row['Diagnosis'], 'Label Categories'
        ].values[0]
    else:
        row['Category'] = pd.NA
    return row


def process_mimic_ecg():
    # Define the file path
    mimic_path = '/biodb/mimic-iv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    df_records = pd.read_csv(os.path.join(mimic_path, 'record_list.csv'))
    df_machine_measurements = pd.read_csv(os.path.join(mimic_path, 'machine_measurements.csv'))
    df_merged = pd.merge(df_records, df_machine_measurements, on='study_id')
    print(df_merged.info())
    file_path = df_records.loc[0, 'path']
    sample_record_path = os.path.join(mimic_path, file_path)
    # Read the record and annotation
    record = wfdb.rdrecord(sample_record_path)
    signal = record.p_signal
    # df_machine_measurements[(df_machine_measurements['subject_id'] == df_records['subject_id']) &
    # (df_machine_measurements['study_id'] == df_records['study_id'])]
    print(df_merged.shape)
    # Print some information about the record
    print(f"Signal shape: {record.p_signal.shape}")
    print(f"Sampling frequency: {record.fs}")
    print(f"Signal channels: {record.sig_name}")
    exit()
    # Plot the ECG signal
    wfdb.plot_wfdb(record=record, figsize=(24, 18), title='MIMIC-IV ECG example', ecg_grids='all')
    plt.savefig('./mimic_iv/plots/sample_ecg.png')

    df_reports = df_machine_measurements[df_machine_measurements.columns[4:22]]
    df_merged['Diagnosis'] = df_reports.apply(
        lambda row: ','.join(v.strip(' .') for v in row if pd.notna(v)),
        axis=1
    )
    print(df_merged['Diagnosis'].shape)
    df_diag = pd.read_excel('./mimic_iv/labels_count_with_categories.xlsx')
    print(df_diag.columns)
    df_merged['Category'] = pd.Series()
    df_merged = df_merged.apply(get_label_categories, df=df_diag, axis=1)
    df_merged.to_csv('./mimic_iv/signals_with_reports.csv', index=False)
    exit()
    print(df_merged.shape)
    df_merged = df_merged[df_merged['Category'].notna()]
    print(df_merged.shape)
    get_labels(np.array(df_merged['Category']))
    exit()
    grouped = df_reports.groupby('Diagnosis').size().reset_index(name='Count')
    df_count = grouped[grouped['Count'] > 100]
    # df_count.to_csv('./mimic_iv/labels_count.csv', index=False)
    # Split each cell by commas and aggregate all values
    all_values = ','.join(grouped['Diagnosis']).split(',')
    # Get unique values
    unique_values = set(all_values)
    print("Unique values: \n", len(unique_values))


def get_labels(label_categories):
    # df_diag = pd.read_excel('./mimic_iv/labels_count_with_categories.xlsx')
    # print(df_diag.head())
    # label_categories = np.array(df_diag['Label Categories'])
    label_list = [str(item) for item in label_categories if item is not np.nan]
    split_labels = [str(item).strip() for sublist in label_list for item in sublist.split(',')]
    unique_labels = sorted(set(split_labels))
    return unique_labels


def get_categories(row, uq_labels, mimic_path):
    file_path = row['path']
    record_path = os.path.join(mimic_path, file_path)
    record = wfdb.rdrecord(record_path)
    zero_array = np.zeros(len(uq_labels), dtype=np.int64)
    diag_labels = str(row['Category']).split(',')
    for diag in diag_labels:
        diag = diag.strip()
        if diag in uq_labels:
            idx = uq_labels.index(diag)
            zero_array[idx] = 1
    return np.transpose(record.p_signal), zero_array


def format_label_categories():
    start_time = time.time()
    df = pd.read_csv('./mimic_iv/files/signals_with_reports.csv')
    df.drop(['ecg_time_x', 'cart_id', 'ecg_time_y', 'report_0', 'report_1', 'report_2', 'report_3', 'report_4',
             'report_5', 'report_6', 'report_7', 'report_8', 'report_9', 'report_10', 'report_11', 'report_12',
             'report_13', 'report_14', 'report_15', 'report_16', 'report_17', 'bandwidth', 'filtering',
             'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis'],
            axis=1, inplace=True)
    df = df[df['Category'].notna()]
    mimic_path = '/biodb/mimic-iv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'

    # 78 unique label combo and total 24 unique labels
    df = df.apply(lambda x: process_categories(x), axis=1)
    df['Category'].replace(
        {'Sinus Bradycardia': 'Sinus Bradycardia, Normal',
         'Sinus Tachycardia': 'Sinus Tachycardia, Normal',
         'Sinus Arrhythmia': 'Sinus Arrhythmia, Normal'}, inplace=True)

    grouped = df.groupby('Category').size().reset_index(name='Count')
    # only consider conditions with atleast 500 ecgs
    high_samples = grouped[grouped['Count'] > 500]
    high_samples.drop(13, inplace=True)

    # 37 label categories
    df_high_samples = df[df['Category'].isin(high_samples['Category'])]

    # get unique list of labels to create label array for the model
    # currently 20 unique labels i.e. length of the label array is 20
    unique_labels = get_labels(np.array(df_high_samples['Category']))
    np.savetxt('./mimic_iv/files/label_names_sequence.txt', unique_labels, fmt='%s')
    print(unique_labels)

    # Create empty 3D arrays to hold all signals and labels
    df_high_samples = df_high_samples.reset_index(drop=True)
    print(df_high_samples.shape)
    signals_array = np.empty((len(df_high_samples), 12, 5000), dtype=np.float64)
    labels_array = np.empty((len(df_high_samples), 20), dtype=np.int64)

    for idx, row in df_high_samples.iterrows():
        signal, label = get_categories(row, uq_labels=unique_labels, mimic_path=mimic_path)
        signals_array[idx] = signal
        labels_array[idx] = label

    print(signals_array.shape, labels_array.shape)

    np.save('./mimic_iv/processed_data/data/mimic_all_data.npy', signals_array)
    np.save('./mimic_iv/processed_data/labels/mimic_all_labels.npy', labels_array)

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time = (elapsed_time * (len(df_high_samples) / 1000)) / 3600
    print(f"Time taken for {len(df_high_samples)} records: {total_time} hours")
    print('Elapsed time in seconds: ', elapsed_time)


def process_categories(row):
    if row['Category'] == 'Sinus Bradycardia':
        row['Labels'] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif row['Category'] == 'Sinus Tachycardia':
        row['Labels'] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif row['Category'] == 'Sinus Arrhythmia':
        row['Labels'] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    return row


def downsample():
    signals = np.load('processed_data/data/mimic_all_data.npy', mmap_mode='r')
    labels = np.load('processed_data/labels/mimic_all_labels.npy', mmap_mode='r')
    print(signals.shape)

    num_samples = signals.shape[0]

    downsampled_signals = np.empty((num_samples, 12, 1000), dtype=np.float64)
    print('Empty', downsampled_signals.shape)

    # Use ThreadPoolExecutor for parallel processing
    def process_signal(i):
        downsampled_signals[i, :, :] = downsample_ecg(signals[i], 1000)

    with ThreadPoolExecutor() as executor:
        executor.map(process_signal, range(num_samples))

    print('Downsampled', downsampled_signals.shape)
    print(labels.shape)

    clean_ecgs(downsampled_signals, labels, sampling_rate=100)


def multi_label_data_split():
    signals = np.load('./mimic_iv/processed_data/data/mimic_all_data_cleaned.npy', mmap_mode='r')
    labels = np.load('./mimic_iv/processed_data/labels/mimic_all_labels_cleaned.npy', mmap_mode='r')

    print(signals.shape, labels.shape)
    print(signals[:2])
    print(labels[:2])

    # Initialize MultilabelStratifiedKFold for train-test split
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize arrays to store the final splits
    X_train_val, y_train_val, X_test, y_test = None, None, None, None

    # Generate the train+val vs. test split indices
    for train_index, test_index in mskf.split(signals, labels):
        X_train_val, X_test = signals[train_index], signals[test_index]
        y_train_val, y_test = labels[train_index], labels[test_index]
        break  # Use only the first split

    print(X_train_val.shape, X_test.shape)

    # Initialize arrays to store the train and validation splits
    X_train, y_train, X_val, y_val = None, None, None, None

    # Generate the train vs. val split indices
    for t_index, v_index in mskf.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[t_index], X_train_val[v_index]
        y_train, y_val = y_train_val[t_index], y_train_val[v_index]
        break  # Use only the first split

    print(X_train.shape, X_val.shape)

    # save the train and test data and labels in numpy format for the model
    np.save('./mimic_iv/processed_data/data/mimic_train_data_cleaned.npy', X_train)
    np.save('./mimic_iv/processed_data/data/mimic_test_data_cleaned.npy', X_test)
    np.save('./mimic_iv/processed_data/data/mimic_val_data_cleaned.npy', X_val)
    np.save('./mimic_iv/processed_data/labels/mimic_train_labels_cleaned.npy', y_train)
    np.save('./mimic_iv/processed_data/labels/mimic_test_labels_cleaned.npy', y_test)
    np.save('./mimic_iv/processed_data/labels/mimic_val_labels_cleaned.npy', y_val)


def clean_ecgs(signals, labels, sampling_rate, verbose=False):
    print(signals.shape, labels.shape)

    def is_ecg_signal(signal, sampling_rate):
        try:
            # Process the signal using NeuroKit's ECG pipeline
            _, info = nk.ecg_peaks(signal, sampling_rate)
            return len(info['ECG_R_Peaks']) > 0  # True if R-peaks detected
        except Exception as e:
            if verbose:
                print(f"Processing error: {e}")
            return False

    # Process each recording
    all_ecgs, all_labels = [], []
    for idx, data in enumerate(signals):
        if verbose:
            print(f"Processing ECG recording {idx + 1}/{len(signals)}")
        ecg_12_lead = []

        try:
            for i, lead in enumerate(data):
                # Clean the lead using the specified method
                invert_lead, _ = nk.ecg_invert(lead, sampling_rate)
                cleaned_lead = nk.ecg_clean(invert_lead, sampling_rate, method="neurokit")

                # Check if this lead is a valid ECG signal
                if is_ecg_signal(cleaned_lead, sampling_rate):
                    ecg_12_lead.append(cleaned_lead)
                else:
                    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                    axes[0].plot(lead)
                    axes[0].set_title('ECG signal Before Cleaning')
                    axes[1].plot(cleaned_lead)
                    axes[1].set_title('ECG signal After Cleaning')
                    plt.savefig(f'temp_plots/final_cleaned_ecg_{idx}_{i}.png')
                    plt.close()

                    print(idx)

                    raise ValueError(f"Invalid ECG signal in lead {len(ecg_12_lead) + 1}")

            all_ecgs.append(ecg_12_lead)
            all_labels.append(labels[idx])

        except Exception as e:
            print(f"Skipping ECG recording {idx} due to error: {e}")
            continue

    all_ecgs = np.array(all_ecgs)
    all_labels = np.array(all_labels)

    print(all_ecgs.shape, all_labels.shape)

    np.save('processed_data/data/mimic_all_data_ds_cleaned.npy', all_ecgs)
    np.save('processed_data/labels/mimic_all_labels_ds_cleaned.npy', all_labels)


def extract_ecg_subset():
    signals = np.load('mimic_iv/processed_data/data/mimic_all_data_ds_cleaned.npy', mmap_mode='r')
    labels = np.load('mimic_iv/processed_data/labels/mimic_all_labels_ds_cleaned.npy', mmap_mode='r')

    # Create a data subset with 20,000 ecgs of label - [Sinus Rhythm, Normal]

    print(signals.shape, labels.shape)
    normal_ecg_subset, normal_labels_subset = [], []
    for i, label in enumerate(labels):
        label = ', '.join(map(str, label))
        label_list = [int(x.strip()) for x in label.split(',')]
        if label_list == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
            normal_ecg_subset.append(signals[i])
            normal_labels_subset.append(labels[i])
        if len(normal_ecg_subset) == 25000:
            break

    normal_ecg_array = np.array(normal_ecg_subset)
    normal_labels_array = np.array(normal_labels_subset)

    non_ecg_indices = filter_ecgs(normal_ecg_array)
    print(len(non_ecg_indices))

    normal_ecg_array = np.delete(normal_ecg_array, non_ecg_indices, axis=0)
    normal_labels_array = np.delete(normal_labels_array, non_ecg_indices, axis=0)

    print(normal_ecg_array.shape, normal_labels_array.shape)

    # 70/30 split into train and temp data
    train_data, temp_data, train_labels, temp_labels = train_test_split(normal_ecg_array, normal_labels_array,
                                                                        test_size=0.30, random_state=42)
    # Further split temp data into 50/50 for validation and test datasets
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels,
                                                                    test_size=0.50, random_state=42)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(val_data[10, 5, :])

    # Calculate min and max on the training data (lead-wise normalization)
    train_min = np.min(train_data, axis=(0, 2), keepdims=True)  # Min for each lead
    train_max = np.max(train_data, axis=(0, 2), keepdims=True)  # Max for each lead

    np.save('mimic_iv/processed_data/subset/mimic_train_min.npy', train_min)
    np.save('mimic_iv/processed_data/subset/mimic_train_max.npy', train_max)

    exit()

    # Normalize the training, validation, and test sets using training min/max
    def min_max_normalize(data, min_val, max_val):
        normalized_data = (data - min_val) / (max_val - min_val)
        return np.clip(normalized_data, 0, 1)

    # Apply normalization lead-wise for all splits
    train_data_normalized = min_max_normalize(train_data, train_min, train_max)
    val_data_normalized = min_max_normalize(val_data, train_min, train_max)
    test_data_normalized = min_max_normalize(test_data, train_min, train_max)

    # Print shapes to verify
    print("Train Data Normalized Shape:", train_data_normalized.shape)
    print("Validation Data Normalized Shape:", val_data_normalized.shape)
    print("Test Data Normalized Shape:", test_data_normalized.shape)

    # Check the normalized data range
    print("Training Data Range:", np.min(train_data_normalized), np.max(train_data_normalized))
    print("Validation Data Range:", np.min(val_data_normalized), np.max(val_data_normalized))
    print("Test Data Range:", np.min(test_data_normalized), np.max(test_data_normalized))

    axs[1].plot(val_data_normalized[10, 5, :])
    plt.savefig(f'mimic_iv/plots/val_data_normalized_{5}_{10}.png')

    np.save('mimic_iv/processed_data/subset/mimic_train_data_normalized_subset.npy', train_data_normalized)
    np.save('mimic_iv/processed_data/subset/mimic_test_data_normalized_subset.npy', test_data_normalized)
    np.save('mimic_iv/processed_data/subset/mimic_val_data_normalized_subset.npy', val_data_normalized)
    np.save('mimic_iv/processed_data/subset/mimic_train_labels_normalized_subset.npy', train_labels)
    np.save('mimic_iv/processed_data/subset/mimic_test_labels_normalized_subset.npy', test_labels)
    np.save('mimic_iv/processed_data/subset/mimic_val_labels_normalized_subset.npy', val_labels)


def filter_ecgs(input_data):
    import antropy as ant

    non_ecgs = []
    template_1 = input_data[8, 1, :]
    template_2 = input_data[47, 1, :]
    for i, data in enumerate(input_data):
        signal = data[1]
        samp_entropy = ant.sample_entropy(signal, order=2, metric='chebyshev')
        corr1 = np.correlate(signal - np.mean(signal), template_1 - np.mean(template_1), mode='full')
        corr2 = np.correlate(signal - np.mean(signal), template_2 - np.mean(template_2), mode='full')
        norm_factor1 = len(signal) * np.std(signal) * np.std(template_1)
        norm_factor2 = len(signal) * np.std(signal) * np.std(template_2)
        corr1 = corr1 / norm_factor1
        corr2 = corr2 / norm_factor2
        # Define a threshold for correlation and sample entropy
        if (max(corr1) > 0.12 or max(corr2) > 0.12) and (samp_entropy < 0.5):
            continue
        else:
            non_ecgs.append(i)

    return non_ecgs


def ecg_quality_check():
    train_data = np.load('./mimic_iv/processed_data/subset/mimic_train_data_normalized_subset.npy')
    non_ecg_indices = filter_ecgs(train_data)
    train_data = train_data[non_ecg_indices[:15], :, :]

    plt.figure(figsize=(12, 5))

    for i, d in enumerate(train_data):
        plt.figure(figsize=(12, 5))
        plt.plot(train_data[i, 0, :])
        plt.savefig(f'./mimic_iv/plots/quality_check_4/{i}.png')
        plt.show()
        plt.clf()


extract_ecg_subset()
