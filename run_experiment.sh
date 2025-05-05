#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=nlp-final
#SBATCH --output=logs/experiment_output_%j.out
#SBATCH --mem=16G

echo "✅ SLURM job started: $(date)" 
hostname
module load anaconda


echo "✅ Python about to run: $(date)"
/home/cs601-lgeer3/scr4-cs601-dkhasha1/cs601-lgeer3/nlp-final/env/bin/python run_experiment.py --dataset ignored --batch_size 64 --hidden_dim 384 --hidden_layers 6 --block_size 256 --epochs 5 --n_head 6 --beta 0.3 --learning_rate 1e-4 --gradient_accumulation 1 --save_model --save_path ./checkpoints/
echo "✅ Python finished: $(date)";


