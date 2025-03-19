# ECG Synthesis

Based on the model here - https://github.com/AI4HealthUOL/SSSD-ECG

This repository provides a script to generate synthetic 12-lead ECGs for 2 labels i.e. healthy and Atrial Fibrillation, using the SSSD_ECG model. The script leverages a pre-trained model checkpoint and configuration files for customization.

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

## **Fine-Tuning the Pre-Trained ECG Model on a Custom Dataset**  

To apply the pre-trained ECG model to a custom dataset and generate Healthy and AFib ECGs, follow these steps:

**1. Prepare Your Custom Dataset**  
Ensure your dataset is formatted correctly before training:  

- **Dataset Shape:**  
  - Create a NumPy array of ECGs with shape **(N, 12, 1000)**, where:  
    - **N** = Number of ECG samples  
    - **12** = Total number of leads  
    - **1000** = Total data points in each ECG sample  

- **Labels Shape:**  
  - Create a corresponding label NumPy array with shape **(N, 20)**.  
  - Labels should be one-hot encoded as follows:  
    - **Healthy ECGs:**  
      ```plaintext
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
      ```
    - **AFib ECGs:**  
      ```plaintext
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      ```

- **Preprocessing Steps:**  
  - **Filtering:** Remove noise and baseline wandering using the `ecg_clean` function from **Neurokit2**.  
  - **Sampling Rate:** Ensure the ECGs have a sample rate of **100Hz**.  
  - **Normalization:** Apply **min-max normalization** lead-wise.  

- **Save Format:**  
  - Save both **dataset** and **labels** as NumPy arrays (`.npy` format).


**2. Set Up the Environment** 
Make sure you have all the dependencies installed and if using virtual environment, make sure it is activated.

**3. Load the Pre-Trained Model**
   - Add the path to the pre-trained model in the config JSON file **SSSD_ECG_MIMIC.json** at the location - **"/train_config/ckpt_path"**. 
   - In the config file, modify the number of iterations to a value between 200,000 to 300,000 iterations, depending on the size of your dataset.

**4. Load Dataset**
   - Add the data path and labels path in the config file at the location - **"/trainset_config/data_path" and "/trainset_config/labels_path"**
   
**5. Train the Model on Your Dataset**
   - Run the training script:
```bash
   python mimic_iv/train.py -c config/SSSD_ECG_MIMIC.json
```
   - The model checkpoints will be saved in the output directory as given in the config file
   - You can evaluate by generating ECGs from the latest model checkpoints following the same steps as shows in next section.

## Usage for Evaluation

1. Prepare Checkpoint:
   Download the pre-trained model checkpoint to generate ECGs from the specified links below.
      - Only healthy ECGs - https://data.dieterichlab.org/s/rbGRSJgZFQTb8Ha
      - Healthy and AF ECGs - https://data.dieterichlab.org/s/D6HC8xaCrjDx3sJ

2. Setup Configuration:
   Specify the downloaded checkpoint path in the JSON configuration file (config/SSSD_ECG_MIMIC.json).

3. Run the ECG generation script:
    ```bash
   python mimic_iv/inference.py -c config/SSSD_ECG_MIMIC.json -n 50 -l afib
   ```
   **Arguments**:
   - -c, --config: Path to the configuration JSON file. (Default: config/SSSD_ECG_MIMIC.json)
   - -n, --num_samples: Number of ECG samples to generate. (Default: 50)
   - -l, --label_type: Define the type of ECG to generate. Choose between "afib" or "healthy"

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

`--label_type`: (Required) The disease label type for ECG visualization

### Example
`python mimic_iv/visualize.py --batch_idx 0 --sample_idx 5 --all_leads --label_type afib`

