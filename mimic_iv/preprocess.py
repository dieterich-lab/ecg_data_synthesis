import os
import sys
import time
import numpy as np
import wfdb
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import normalize_data

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


def clean_ecgs():
    signals = np.load('./mimic_iv/processed_data/data/mimic_all_data.npy', mmap_mode='r')
    labels = np.load('./mimic_iv/processed_data/labels/mimic_all_labels.npy', mmap_mode='r')

    # Process each recording
    all_ecgs, all_labels = [], []
    for idx, data in enumerate(signals):
        print(f"Processing ECG recording {idx + 1}/{len(signals)}")
        ecg_12_lead = []
        try:
            for lead in data:
                processed_lead = nk.ecg_clean(lead, sampling_rate=500, method="neurokit")
                ecg_12_lead.append(processed_lead)
            all_ecgs.append(ecg_12_lead)
            all_labels.append(labels[idx])
        except:
            print('Something wrong with the ECG. Skipping....')
            continue

    all_ecgs = np.array(all_ecgs)
    all_labels = np.array(all_labels)

    # fig, axes = plt.subplots(2, 1)
    # axes[0].plot(signals[10, 0, :])
    # axes[0].set_title('ECG signal Before Cleaning')
    # axes[1].plot(all_ecgs[10, 0, :])
    # axes[1].set_title('ECG signal After Cleaning')
    # plt.show()

    np.save('./mimic_iv/processed_data/data/mimic_all_data_cleaned.npy', all_ecgs)
    np.save('./mimic_iv/processed_data/labels/mimic_all_labels_cleaned.npy', all_labels)

    print(all_ecgs.shape)
    print(all_labels.shape)


def normalize():
    train_data = np.load('./mimic_iv/processed_data/data/mimic_train_data_cleaned.npy')
    val_data = np.load('./mimic_iv/processed_data/data/mimic_val_data_cleaned.npy')
    test_data = np.load('./mimic_iv/processed_data/data/mimic_test_data_cleaned.npy')
    temp_data = np.load('./mimic_iv/processed_data/data/mimic_all_data_100_cleaned.npy')

    # filtered_data, filtered_labels = remove_nan(data_ptbxl, labels_ptbxl)
    normalized_train_data = normalize_data(train_data)
    normalized_val_data = normalize_data(val_data)
    normalized_test_data = normalize_data(test_data)
    normalized_temp_data = normalize_data(temp_data)

    np.save('./mimic_iv/processed_data/data/mimic_train_data_cleaned_normalized.npy', normalized_train_data)
    np.save('./mimic_iv/processed_data/data/mimic_val_data_cleaned_normalized.npy', normalized_val_data)
    np.save('./mimic_iv/processed_data/data/mimic_test_data_cleaned_normalized.npy', normalized_test_data)
    np.save('./mimic_iv/processed_data/data/mimic_temp_data_cleaned_normalized.npy', normalized_temp_data)


def normalize_temp_data():
    temp_data = np.load('./mimic_iv/processed_data/data/mimic_all_data_100_cleaned.npy')
    normalized_temp_data = normalize_data(temp_data)
    np.save('./mimic_iv/processed_data/data/mimic_100_cleaned_normalized.npy', normalized_temp_data)


normalize_temp_data()
