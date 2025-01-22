#!/bin/bash
#SBATCH --job-name=lora_ft_1gpu_3epo_final         # Job name
#SBATCH --output=LoRA_finetune_%j.log              # Output log file (%j will be replaced with the job ID)
#SBATCH --error=LoRA_finetune_%j.err               # Error log file (%j will be replaced with the job ID)
#SBATCH --partition=GPUQ                           # Partition to run on (GPUQ)
#SBATCH --gres=gpu:1                               # Request 1 GPU
#SBATCH --cpus-per-task=4                          # Request 4 CPU cores
#SBATCH --mem=16G                                  # Request 16GB of RAM
#SBATCH --time=3-00:00:00                          # format: D-HH:MM:SS
#SBATCH --mail-user=md.a.hasan@ntnu.no             # Email notifications
#SBATCH --mail-type=ALL                            # Notify on all events

# Export CUDA paths to avoid conflicts
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export CUDA_HOME=/usr/local/cuda

# Activate conda environment
module load Anaconda3/2023.09-0
conda activate codegen_env

# Move to the submit directory
cd $SLURM_SUBMIT_DIR

# Print SLURM environment information
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"


# Run the LoRA training script
python LoRA_ft_script_final.py
