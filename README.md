# ECG Synthesis

Based on the model here - https://github.com/AI4HealthUOL/SSSD-ECG

This repository provides a script to generate synthetic 12-lead ECGs using the SSSD_ECG model. The script leverages a pre-trained model checkpoint and configuration files for customization.

## Installation

1. Clone the repository:
   ```bash
   git@github.com:dieterich-lab/ecg_data_synthesis.git
   cd ecg_data_synthesis

2. Install dependencies:
   ```bash
   pip install -e .


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
   **Arguments**
   -c, --config: Path to the configuration JSON file. (Default: config/SSSD_ECG_MIMIC.json)
   -n, --num_samples: Number of ECG samples to generate. (Default: 50)

## Outputs
The generated ECG samples are saved in a directory specified in the configuration file under gen_config.output_directory. 
Each batch produces:
   - Generated ECG Data: <iteration>_gen_ecg.npy
   - Labels: <iteration>_labels.npy


       
