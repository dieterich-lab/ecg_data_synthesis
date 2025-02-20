# ECG Synthesis

Based on the model here - https://github.com/AI4HealthUOL/SSSD-ECG

This repository provides a script to generate synthetic 12-lead ECGs using the SSSD_ECG model. The script leverages a pre-trained model checkpoint and configuration files for customization.

## Requirements
- Python 3.10+
- A server with **NVIDIA GPU** support

## Installation

1. Clone the repository:
   ```bash
   git@github.com:dieterich-lab/ecg_data_synthesis.git
   cd ecg_data_synthesis
   
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
   
   Open the virtual environment's activation script: `.venv/bin/activate` and add the following line at the end of the script.
   ```bash
   export PYTHONPATH="$PYTHONPATH:$(pwd)"
   ```
   This will ensure that the PYTHONPATH is set to the current working directory every time the environment is activated.
   
   Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install .


## Usage

1. Prepare Checkpoint:
   Download the pre-trained model checkpoint from here - https://data.dieterichlab.org/s/rbGRSJgZFQTb8Ha

2. Setup Configuration:
   Specify the downloaded checkpoint path in the JSON configuration file (config/SSSD_ECG_MIMIC.json).

3. Run the ECG generation script:
   ```bash
   model_inference
   ```
   or run using python,
    ```bash
   python inference.py -c config/SSSD_ECG_MIMIC.json -n 50
   ```
   **Arguments**:
   - -c, --config: Path to the configuration JSON file. (Default: config/SSSD_ECG_MIMIC.json)
   - -n, --num_samples: Number of ECG samples to generate. (Default: 50)

## Outputs
The generated ECG samples are saved in a directory specified in the configuration file under gen_config.output_directory. 
Each batch produces:
   - Generated ECG Data: `<batch_idx>_gen_ecg.npy` with shape: `(N, 12, 1000)`
   - Labels: `<batch_idx>_labels.npy` with shape: `(N, 20)`

## To Display Generated ECGs (Optional)
Try utilizing the `visualize.py` script under `mimic_iv` folder by modifying it according to your setup.
This script is used to visualize ECG signals from generated numpy (.npy) files. The script loads ECG data and associated labels, then plots the ECG waveform for a specified sample.

### **Usage**
`python mimic_iv/visualize.py --batch_idx <BATCH_INDEX> --sample_idx <SAMPLE_INDEX> [--all_leads]`

### Arguments

`--batch_idx`: (Required) The batch index of the ECG data file.

`--sample_idx`: (Required) The sample index to visualize within the batch.

`--all_leads`: (Optional) If included, plots all available ECG leads else plots lead I only

### Example
`python mimic_iv/visualize.py --batch_idx 0 --sample_idx 5 --all_leads`








       
