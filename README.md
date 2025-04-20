# nlp-final
NLP: Self Supervised Model final project. Objective - achieving lowest complexity for LM on 16GB MIG GPU.
## Setup Instructions

1. Clone the repository
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
2. Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate nlp_final
    ```
3. Install any additional packages (if needed)

## Run Experiments

1. Submit a job to Rockfish:
    ```bash
    sbatch run_experiment.sh
    ```

2. Monitor logs in the `/logs/` directory:
    ```bash
    tail -f logs/experiment_output_<jobid>.out
    ```

## Project Structure

- `data_preprocessing/` — Data loading and tokenization
- `model/` — Model architectures
- `training/` — Training and validation loops
- `experiments/` — Scripts that run experiments
- `logs/` — SLURM output and error logs
