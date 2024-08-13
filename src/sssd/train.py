import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mimic_iv.utils import remove_nan, normalize_data
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          batch_size):
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
        filename='../../mimic_iv/sssd_mimic_training.log',
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
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config)

    # Log the total of trainable parameters
    model_size = sum(t.numel() for t in net.parameters())
    logger.info(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")

    # Use DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net = net.cuda()

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
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # here = os.path.dirname(os.path.abspath(__file__))
    # filename_train = os.path.join(here, 'ptbxl_train_data.npy')
    # filename_train_label = os.path.join(here, 'ptbxl_train_labels.npy')
    # data_ptbxl = np.load(filename_train)
    # labels_ptbxl = np.load(filename_train_label)

    train_data = np.load('../../mimic_iv/processed_data/data/mimic_train_data_cleaned.npy')
    train_labels = np.load('../../mimic_iv/processed_data/labels/mimic_train_labels_cleaned.npy')
    val_data = np.load('../../mimic_iv/processed_data/data/mimic_val_data_cleaned.npy')
    val_labels = np.load('../../mimic_iv/processed_data/labels/mimic_val_labels_cleaned.npy')

    # filtered_data, filtered_labels = remove_nan(data_ptbxl, labels_ptbxl)
    normalized_train_data = normalize_data(train_data)
    normalized_val_data = normalize_data(val_data)

    train_dataset, val_dataset = [], []
    for i in range(normalized_train_data.shape[0]):
        train_dataset.append([normalized_train_data[i], train_labels[i]])

    for i in range(normalized_val_data.shape[0]):
        val_dataset.append([normalized_val_data[i], val_labels[i]])

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    # ['I','II','V1','V2','V3','V4','V5','V6','III','AVR','AVL','AVF'] for processed ptbxl dataset
    # index_8 = torch.tensor([0, 2, 3, 4, 5, 6, 7, 11])
    # index_4 = torch.tensor([1, 8, 9, 10])

    # ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] for mimic-iv dataset
    index_8 = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11])
    index_4 = torch.tensor([1, 2, 3, 5])

    # training
    n_iter = ckpt_iter + 1

    logger.info('Training started')

    train_loss_epoch = []
    while n_iter < n_iters + 2:
        logger.info(f'Epoch: {n_iter}')
        train_loss_batch = []
        for step, (data, label) in enumerate(train_loader):

            data = torch.index_select(data, 1, index_8).float().cuda()
            label = label.float().cuda()

            # back-propagation
            optimizer.zero_grad()

            X = data, label

            loss = training_loss_label(net, nn.MSELoss(), X, diffusion_hyperparams)

            loss.backward()
            optimizer.step()

            print(loss.item())
            train_loss_batch.append(loss.item())
            if n_iter % iters_per_logging == 0:
                logger.info("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                logger.info('model at iteration %s is saved' % n_iter)

            n_iter += 1

        avg_train_loss = sum(train_loss_batch) / len(train_loss_batch)
        train_loss_epoch.append(avg_train_loss)
        logger.info(f'Training loss for the current epoch: {avg_train_loss}')

    plt.plot(train_loss_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss for SSSD-ECG model with MIMIC-IV ECG data')
    plt.savefig('../../mimic_iv/plots/training_loss.png')

    logger.info('Training completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/SSSD_ECG_MIMIC_temp.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    train(**train_config)
