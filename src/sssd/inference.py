import os
import argparse
import json
import numpy as np
import torch
import sys

from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams
from models.SSSD_ECG import SSSD_ECG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def generate_four_leads(tensor):
    leadI = tensor[:, 0, :].unsqueeze(1)
    leadschest = tensor[:, 1:7, :]
    leadavf = tensor[:, 7, :].unsqueeze(1)

    leadII = (0.5 * leadI) + leadavf

    leadIII = -(0.5 * leadI) + leadavf
    leadavr = -(0.75 * leadI) - (0.5 * leadavf)
    leadavl = (0.75 * leadI) - (0.5 * leadavf)

    leads12 = torch.cat([leadI, leadII, leadIII, leadavr, leadavl, leadavf, leadschest], dim=1)

    return leads12


def generate(output_directory,
             num_samples,
             ckpt_path,
             ckpt_iter, ):
    """
    Generate processed_data based on ground truth

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    """

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"],
                                           diffusion_config["T"],
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, 'generated_ecgs')
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config).cuda()
    net = torch.nn.DataParallel(net)
    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
        print(ckpt_iter)
    model_path = os.path.join(ckpt_path, 'best_model_{}.pkl'.format(ckpt_iter))
    try:
        print("Loading checkpoint from {}".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except BaseException as e:
        raise Exception('No valid model found - {}'.format(e))

    test_data = np.load('./mimic_iv/processed_data/data/mimic_test_data_cleaned_normalized.npy')
    test_labels = np.load('./mimic_iv/processed_data/labels/mimic_test_labels_cleaned.npy')

    test_dataset = []
    for i in range(test_data.shape[0]):
        test_dataset.append([test_data[i], test_labels[i]])
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=10,
                                              drop_last=True, num_workers=4)

    for i, batch in enumerate(test_loader):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        real_ecg_batch = batch[0]
        label_batch = batch[1]
        cond = label_batch.cuda().float()

        generated_ecg = sampling_label(net, (label_batch.shape[0], 8, trainset_config["segment_length"]),
                                       diffusion_hyperparams, cond=cond)
        generated_ecg12 = generate_four_leads(generated_ecg)

        end.record()
        torch.cuda.synchronize()
        print('generated {} utterances of random_digit at iteration {} in {} seconds'
              .format(num_samples, i, int(start.elapsed_time(end) / 1000)))

        outfile = f'{i}_gen_ecg.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, generated_ecg12.detach().cpu().numpy())
        print('saved generated samples at iteration %s' % i)

        outfile = f'{i}_labels.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, cond.detach().cpu().numpy())
        print('saved the conditioned labels at iteration %s' % i)

        outfile = f'{i}_real_ecg.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, real_ecg_batch.detach().cpu().numpy())
        print('saved real samples at iteration %s' % i)

        if i == 2:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./src/sssd/config/SSSD_ECG_MIMIC.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=50,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

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

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples, )
