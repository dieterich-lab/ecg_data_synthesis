import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams


def setup_logger(log_file):
    """Sets up the logger for training process."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('SSSD-MIMIC Training')


def create_output_directory(base_dir, model_config, diffusion_config):
    """Creates the output directory for saving checkpoints."""
    subdir = f"ch{model_config['res_channels']}_T{diffusion_config['T']}_betaT{diffusion_config['beta_T']}"
    output_directory = os.path.join(base_dir, subdir)
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    return output_directory


def load_training_data(data_path, labels_path):
    """Loads training data and labels."""
    data = np.load(data_path)
    labels = np.load(labels_path)
    print(f"Data shapes: {data.shape}, {labels.shape}")
    train_dataset = [[data[i], labels[i]] for i in range(len(data))]
    return train_dataset


def initialize_model(model_config, learning_rate):
    """Initializes the model and optimizer."""
    sssd_model = SSSD_ECG(**model_config)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        sssd_model = nn.DataParallel(sssd_model)
    sssd_model = sssd_model.cuda()
    optim = torch.optim.AdamW(sssd_model.parameters(), lr=learning_rate)
    return sssd_model, optim


def load_checkpoint(output_directory, sssd_model, optim, ckpt_iter):
    """Loads model and optimizer state from checkpoint, if available."""
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            ckpt_path = os.path.join(output_directory, f"{ckpt_iter}.pkl")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            sssd_model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Successfully loaded model at iteration {ckpt_iter}")
        except:
            ckpt_iter = -1
            print("'No valid checkpoint model found, start training from initialization.")
    else:
        ckpt_iter = -1
        print("No valid checkpoint model found, start training from initialization.")
    return ckpt_iter


def train_model(train_config, train_dataset, sssd_model, optimizer, logger):
    """Trains the SSSD-ECG model for the given number of iterations."""
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_config['batch_size'], shuffle=True, drop_last=True
    )

    # Define the indices for ECG channels
    # ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] for mimic-iv dataset
    index_8 = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11])
    index_4 = torch.tensor([1, 2, 3, 5])

    n_iter = train_config['ckpt_iter'] + 1
    iter_loss = []

    logger.info("Training started")
    while n_iter < train_config['n_iters'] + 1:
        for audio, label in train_loader:

            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()

            optimizer.zero_grad()
            loss = training_loss_label(sssd_model, nn.MSELoss(), (audio, label), diffusion_hyperparams)
            loss.backward()
            optimizer.step()

            iter_loss.append(loss.item())
            if n_iter % train_config['iters_per_logging'] == 0:
                logger.info(f"Iteration {n_iter}: Loss = {loss.item()}")

            if n_iter > 0 and n_iter % train_config['iters_per_ckpt'] == 0:
                save_path = os.path.join(train_config['output_directory'], f"{n_iter}.pkl")
                torch.save({
                    'model_state_dict': sssd_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, save_path)
                print(f"Checkpoint saved at iteration {n_iter}")

            n_iter += 1

    plt.figure(figsize=(10, 5))
    plt.plot(iter_loss, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss per Iteration for MIMIC-IV ECGs')
    plt.legend()
    plt.savefig('mimic_iv/train_loss_iteration_mimic.png')
    logger.info('Training completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='src/sssd/config/SSSD_ECG_MIMIC.json',
                        required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    # Load configurations
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    # Initialize components
    logger = setup_logger("mimic_iv/sssd_training_mimic.log")
    output_dir = create_output_directory(
        config['train_config']['output_directory'], config['wavenet_config'], config['diffusion_config']
    )
    train_data = load_training_data(config['trainset_config']['data_path'], config['trainset_config']['labels_path'])

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**config['diffusion_config'])

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    model, optimizer = initialize_model(config['wavenet_config'], config['train_config']['learning_rate'])

    # Train the model
    config['train_config']['output_directory'] = output_dir
    config['train_config']['ckpt_iter'] = load_checkpoint(
        output_dir, model, optimizer, config['train_config']['ckpt_iter']
    )
    train_model(config['train_config'], train_data, model, optimizer, logger)