import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to sys.path if not already added
if project_root not in sys.path:
    sys.path.append(project_root)

from models.SSSD_ECG import SSSD_ECG
from utils.util import print_size, sampling_label, calc_diffusion_hyperparams


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_four_leads(tensor):
    """
    Generates 12-lead ECG from an input tensor.
    """
    leadI = tensor[:, 0, :].unsqueeze(1)
    leadavf = tensor[:, 7, :].unsqueeze(1)
    leadII = (0.5 * leadI) + leadavf
    leadIII = -(0.5 * leadI) + leadavf
    leadavr = -(0.75 * leadI) - (0.5 * leadavf)
    leadavl = (0.75 * leadI) - (0.5 * leadavf)
    leadschest = tensor[:, 1:7, :]
    return torch.cat([leadI, leadII, leadIII, leadavr, leadavl, leadavf, leadschest], dim=1)


def setup_output_directory(base_path, label_type):
    """
    Creates and returns a directory for saving generated ECGs.
    """
    output_dir = os.path.join(base_path, f"generated_ecgs_{label_type}")
    os.makedirs(output_dir, exist_ok=True)
    os.chmod(output_dir, 0o775)
    logger.info(f"Output directory created: {output_dir}")
    return output_dir


def load_model(ckpt_path, model_config):
    """
    Loads and returns the model from the specified checkpoint.
    """
    net = SSSD_ECG(**model_config).cuda()
    print_size(net)

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


def create_labels(num_samples, label_type):
    """
    Generates a 2D NumPy array of labels for either healthy or AFib ECGs.

    Args:
        num_samples (int): Number of samples to generate.
        label_type (str): Either 'healthy' or 'afib'.

    Returns:
        np.ndarray: A 2D array of labels.
    """
    if label_type == "healthy":
        base_pattern = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # Healthy: Sinus rhythm + Normal diagnosis
    elif label_type == "afib":
        base_pattern = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # AFib

    labels_array = np.tile(base_pattern, (num_samples, 1))
    return labels_array



def save_samples(output_directory, iteration, generated_ecg12, cond):
    """
    Saves generated ECGs and labels to disk.
    """
    for name, data in zip(
            [f'{iteration}_gen_ecg.npy', f'{iteration}_labels.npy'],
            [generated_ecg12, cond]
    ):
        np.save(os.path.join(output_directory, name), data.detach().cpu().numpy())
        logger.info(f"Saved {name} at iteration {iteration}")


def generate(output_directory, num_samples, ckpt_path, model_config, label_type):
    """
    Generates and saves ECG samples based on the specified label type.

    Args:
        output_directory (str): Directory to save ECGs.
        num_samples (int): Number of ECGs to generate.
        ckpt_path (str): Model checkpoint path.
        model_config (dict): Model configuration.
        label_type (str): "healthy" or "afib".
    """
    net = load_model(ckpt_path, model_config)

    if config["gen_config"]["labels_path"]:
        labels = np.load(config["gen_config"]["labels_path"])
        print('Using the labels from config file')
    else:
        labels = create_labels(num_samples, label_type)
        print('Created own labels from the base pattern')

    labels_loader = torch.utils.data.DataLoader(labels, batch_size=10, shuffle=False)

    for i, label_batch in enumerate(labels_loader):
        cond = label_batch.cuda().float()
        generated_ecg = sampling_label(
            net,
            (label_batch.shape[0], 8, config["trainset_config"]["segment_length"]),
            diffusion_hyperparams,
            cond=cond
        )
        generated_ecg12 = generate_four_leads(generated_ecg)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        save_samples(output_directory, i, generated_ecg12, cond)

        end.record()
        torch.cuda.synchronize()

        elapsed_time = start.elapsed_time(end) / 1000
        logger.info(f'Generated {label_type} batch {i + 1}/{(num_samples + 9) // 10} in {elapsed_time:.2f} seconds.')

        if i == 2:
            break

    print('Inference Completed!')


def main():
    print("Running model inference script!")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG_MIMIC.json',
                        help='Path to the configuration JSON file')
    parser.add_argument('-n', '--num_samples', type=int, default=50, help='Number of ECGs to generate')
    parser.add_argument('-l', '--label_type', type=str, choices=['healthy', 'afib'], default='healthy',
                        help='Type of ECG to generate: "healthy" or "afib"')
    args = parser.parse_args()
    print(args)

    print(os.getcwd())

    global config
    with open(args.config) as f:
        config = json.load(f)

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    output_dir = setup_output_directory(
        config['gen_config']['output_directory'], label_type=args.label_type
    )

    generate(
        output_directory=output_dir,
        num_samples=args.num_samples,
        ckpt_path=config['gen_config']['ckpt_path'],
        model_config=config['wavenet_config'],
        label_type=args.label_type
    )


if __name__ == "__main__":
    main()
