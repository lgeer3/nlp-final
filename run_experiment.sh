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
nvidia-smi
module load anaconda
source ~/nlp_envs/nlp_final_local/bin/activate

echo "✅ Python about to run: $(date)"
python run_experiment.py --dataset dogtooth/default_project_dev_test --batch_size 64 --vocab_trimming --vocab_size 10000 --hidden_dim 256 --hidden_layers 5 --block_size 64 --epochs 2 --learning_rate 1e-5 --save_model --save_path ./checkpoints/
echo "✅ Python finished: $(date)";


