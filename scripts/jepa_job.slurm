#!/bin/bash
#SBATCH --job-name=msanocki_scaling_crl     # Job name
#SBATCH --qos=msanocki_common
#SBATCH --partition=common
#SBATCH --nodelist steven
#SBATCH --gres=gpu:titanv:1
#SBATCH --mem-per-cpu=8G                 # Memory per CPU core
#SBATCH --time 20:00:00                   # Time limit (hh:mm:ss)
#SBATCH --output=slurm_logs/slurm-%j.out       # Output file (%j = job ID)

module purge

uv run train.py --env_id "ant_big_maze" --eval_env_id "ant_big_maze_eval" --num_epochs 50 --total_env_steps 50000000 --critic_depth 32 --actor_depth 32 --actor_skip_connections 4 --critic_skip_connections 4 --batch_size 64 --num_envs 128 --vis_length 1000 --use_jepa 1 --jepa_continue_training 1 --save_buffer 1
