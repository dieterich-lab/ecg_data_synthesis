#!/bin/bash

##SBATCH --gres=gpu:turing:1
#SBATCH --job-name=infer_mimic
#SBATCH --output=output_infer_mimic_110000.txt
#SBATCH --partition=gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=20G

srun venv/bin/python -u src/sssd/inference.py --config 'src/sssd/config/SSSD_ECG_MIMIC.json'
