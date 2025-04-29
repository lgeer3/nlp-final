#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name="nlp-final"
#SBATCH --output=logs/experiment_output_%j.out
#SBATCH --mem=16G

module load anaconda
conda activate hw7 # activate the Python environment

python run_experiment.py --dataset dogtooth/default_project_dev_test --batch_size 64 --vocab_trimming --vocab_size 8000 --hidden_dim 256 --hidden_layers 4 --block_size 64 --epochs 2 --learning_rate 1e-5 --save_model --save_path ./checkpoints/

