#!/bin/bash
#SBATCH --job-name=msanocki_scaling_crl_encoder     # Job name
#SBATCH --qos=msanocki_common
#SBATCH --partition=common
#SBATCH --nodelist steven
#SBATCH --gres=gpu:titanv:1
#SBATCH --mem-per-cpu=8G                 # Memory per CPU core
#SBATCH --time 15:00:00                   # Time limit (hh:mm:ss)
#SBATCH --output=slurm_logs/slurm-%j.out       # Output file (%j = job ID)

module purge

uv run train_jepa.py --dataset_path transition_datasets/dataset_5M.npz