#!/bin/bash
#SBATCH --job-name=msanocki_scaling_crl     # Job name
#SBATCH --qos=msanocki_a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=8G                 # Memory per CPU core
#SBATCH --time 15:00:00                   # Time limit (hh:mm:ss)
#SBATCH --output=slurm_logs/slurm-%j.out       # Output file (%j = job ID)

module purge

uv run train.py --env_id "ant_big_maze" --eval_env_id "ant_big_maze_eval" --num_epochs 50 --total_env_steps 50000000 --use_jepa 1 --jepa_checkpoint_path "wandb_crl/runs/ant_big_maze_1000_20260610-075422/step_96977920_ep97.pkl" --jepa_continue_training 0 --jepa_use_predictor_representation 1  --critic_depth 16 --save_buffer 0
