import argparse
import json
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams


def load_pretrained_model(logger, ckpt_path):
    net = SSSD_ECG(**model_config).cuda()

    try:
        logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # remove module. prefix from the keys
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        logger.info(f"Successfully loaded the model checkpoint!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise Exception(f"No valid model found - {e}")

    return net


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          batch_size,
          ckpt_path,
          filename_train,
          filename_train_label):
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
        filename='mimic_iv/finetune.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('Fine-tuning SSSD-MIMIC Model to Generate AF ECGs')

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

    # load pre-trained model
    net = load_pretrained_model(logger, ckpt_path)

    # Log the total of trainable parameters
    model_size = sum(t.numel() for t in net.parameters())
    logger.info(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")

    print("Using", torch.cuda.device_count(), "GPUs!")

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

            state_dict = checkpoint['model_state_dict']

            print("Checkpoint keys:", state_dict.keys())
            print("Model keys:", net.state_dict().keys())

            from collections import OrderedDict
            # Add "module." prefix if needed
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k
                new_state_dict[name] = v


            net.load_state_dict(new_state_dict, strict=False)

            print('model load check!')

            # feed model dict and optimizer state
            # net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('optimizer load check!')

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    data = np.load(filename_train)
    labels = np.load(filename_train_label)

    print('Shapes: ', data.shape, labels.shape)

    train_data = []
    for i in range(len(data)):
        train_data.append([data[i], labels[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] for mimic-iv dataset
    index_8 = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11])
    index_4 = torch.tensor([1, 2, 3, 5])

    # training
    n_iter = ckpt_iter + 1

    logger.info('Training started')

    iter_loss = []

    while n_iter < n_iters + 1:
        for audio, label in trainloader:

            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()

            # back-propagation
            optimizer.zero_grad()

            X = audio, label

            loss = training_loss_label(net, nn.MSELoss(), X, diffusion_hyperparams)
            iter_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                logger.info("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1

    plt.figure(figsize=(10, 5))
    plt.plot(iter_loss, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss per Iteration for Fine-tuned AF ECGs')
    plt.legend()
    plt.savefig('mimic_iv/train_loss_iteration_mimic_afib_v2_0.0001_ext.png')

    logger.info('Training completed')


def main():
    print("Running model training script!")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA
    torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG_MIMIC.json',
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

    train(**train_config,
          filename_train=["data_path"],
          filename_train_label=["labels_path"])


if __name__ == "__main__":
    main()
