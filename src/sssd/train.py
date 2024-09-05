import argparse
import json
import logging
import os
import random
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import tensorboard

cudnn.benchmark = True
cudnn.enabled = True

writer = SummaryWriter('./mimic_iv/tensorboard_runs/experiment_1')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def print_gpu_utilization():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


def train(output_dir_orig,
          ckpt_iter,
          epochs_per_ckpt,
          max_epochs,
          learning_rate,
          batch_size,
          patience,
          min_delta):
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint,
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """

    logging.basicConfig(
        filename='./mimic_iv/sssd_mimic_training.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('SSSD-MIMIC Training')

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"],
                                           diffusion_config["T"],
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_dir_orig, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disable to ensure reproducibility

    # Set seed for reproducibility
    init_seed = 42
    set_seed(init_seed)

    class MimicDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label

    def worker_init_fn(worker_id):
        """
        Initialize worker seed for reproducibility.
        """
        np.random.seed(init_seed + worker_id)
        random.seed(init_seed + worker_id)

    # Load training and validation data with labels
    train_data = np.load('./mimic_iv/processed_data/data/mimic_train_data_cleaned_normalized.npy')
    train_labels = np.load('./mimic_iv/processed_data/labels/mimic_train_labels_cleaned.npy')
    val_data = np.load('./mimic_iv/processed_data/data/mimic_val_data_cleaned_normalized.npy')
    val_labels = np.load('./mimic_iv/processed_data/labels/mimic_val_labels_cleaned.npy')

    print_gpu_utilization()

    # Create datasets
    train_dataset = MimicDataset(train_data, train_labels)
    val_dataset = MimicDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset,
                              shuffle=True, batch_size=batch_size, drop_last=True,
                              num_workers=6, worker_init_fn=worker_init_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            shuffle=False, batch_size=batch_size, drop_last=True,
                            num_workers=6, worker_init_fn=worker_init_fn, pin_memory=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config)

    # Log the total of trainable parameters
    model_size = sum(t.numel() for t in net.parameters())
    print(model_size)
    logger.info(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")

    # Use DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, [0, 1])

    net = net.cuda()

    print_gpu_utilization()

    # define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except BaseException as e:
            ckpt_iter = -1
            print(f'No valid checkpoint model found, start training from initialization try - {e}')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] for mimic-iv dataset
    index_8 = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11])
    index_4 = torch.tensor([1, 2, 3, 5])

    # Training phase
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info('Training started')

    train_loss_epoch, val_loss_epoch = [], []
    train_loss_iter, val_loss_iter = [], []
    epoch_times = []

    for epoch in range(max_epochs):
        start_time = time.time()  # Start time of the epoch
        logger.info(f'Epoch: {epoch}')

        # Training phase
        net.train()  # Set model to train mode
        train_loss_batch = []
        accumulation_steps = 4
        for step, (ecg, label) in enumerate(train_loader):
            ecg = torch.index_select(ecg, 1, index_8).float().cuda()
            label = label.float().cuda()

            # back-propagation
            optimizer.zero_grad()
            X = ecg, label
            loss = training_loss_label(net, nn.MSELoss(), X, diffusion_hyperparams)
            logger.info(f"Epoch {epoch} | Step {step}/{len(train_loader) - 1} | Training loss: {loss.item():.6f}")
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()

            train_loss_batch.append(loss.item())
            train_loss_iter.append(loss.item())

            # Log the loss to TensorBoard
            if step % 1000 == 999:  # Log every 1000 iterations
                writer.add_scalar('Training Loss', sum(train_loss_batch)/1000, epoch * len(train_loader) + step)

        avg_train_loss = sum(train_loss_batch) / len(train_loss_batch)
        train_loss_epoch.append(avg_train_loss)
        logger.info(f'Epoch {epoch} | Average Training Loss: {avg_train_loss:.6f}')

        # Validation phase
        net.eval()  # Set model to evaluation mode
        val_loss_batch = []
        with torch.no_grad():  # Disable gradient computation for validation
            for v_step, (v_data, v_label) in enumerate(val_loader):
                v_data = torch.index_select(v_data, 1, index_8).float().cuda()
                v_label = v_label.float().cuda()
                val_X = v_data, v_label

                val_loss = training_loss_label(net, nn.MSELoss(), val_X, diffusion_hyperparams)
                val_loss_batch.append(val_loss.item())
                val_loss_iter.append(val_loss.item())

                logger.info(
                    f'Epoch {epoch} | Validation Step {v_step}/{len(val_loader) - 1} | '
                    f'Validation loss: {val_loss.item():.6f}')

        avg_val_loss = sum(val_loss_batch) / len(val_loss_batch)
        val_loss_epoch.append(avg_val_loss)
        logger.info(f'Epoch {epoch} | Average Validation Loss: {avg_val_loss:.6f}')

        # Log the validation loss to TensorBoard
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)

        # Calculate and log epoch time
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        logger.info(f"Time taken for epoch {epoch}: {epoch_time:.2f} seconds")

        # Early stopping logic based on validation loss
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset counter if improvement found

            # Save the best model
            best_model_path = os.path.join(output_directory, 'best_model_{}.pkl'.format(epoch))
            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       best_model_path)
            logger.info(f'Saved the best model at epoch {epoch} with validation loss: {best_val_loss:.6f}')

        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
            break  # Exit the loop if early stopping is triggered

        # Step the scheduler with the average validation loss
        scheduler.step(avg_val_loss)
        torch.cuda.empty_cache()

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    logger.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

    # Plot loss per iteration
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_iter, label='Training Loss')
    plt.plot(val_loss_iter, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Iteration for MIMIC-IV ECGs')
    plt.legend()
    plt.savefig('./mimic_iv/plots/train_val_loss_iteration.png')
    plt.clf()

    # Plot loss per epoch
    plt.plot(train_loss_epoch, label='Training Loss')
    plt.plot(val_loss_epoch, label='Validation Loss ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per epoch for MIMIC-IV ECGs')
    plt.legend()
    plt.savefig('./mimic_iv/plots/train_val_loss_epoch.png')

    writer.close()
    logger.info('Training completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./src/sssd/config/SSSD_ECG_MIMIC.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    trainset_config = config["trainset_config"]  # to load trainset

    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    model_config = config['wavenet_config']

    train(**train_config)
