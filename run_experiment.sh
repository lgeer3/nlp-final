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

python experiments/run_experiment.py --model \
    --seed \
    --dataset \
    --epochs \
    --batch_size \
    --learning_rate \
    --gradient_accumulation \
    --hidden_dim \
    --hidden_layers \