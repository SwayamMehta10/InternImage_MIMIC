#!/bin/bash
#SBATCH -J eval_internimage
#SBATCH -p public
#SBATCH -q class
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

# Evaluation script for InternImage on MIMIC-CXR test set
# This runs independently from training and can be submitted while training is running

echo "=========================================="
echo "Evaluation job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

# Activate conda environment
source activate INTERNIMAGE_ENV

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Print environment info
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "GPU info:"
nvidia-smi

echo "=========================================="
echo "Starting evaluation on test set..."
echo "=========================================="

# Evaluation command
# Change --split to 'val' or 'train' if you want to evaluate on those splits instead
python eval_checkpoint.py \
    --cfg configs/internimage_b_mimic_cxr_224.yaml \
    --data-path /scratch/smehta90/mimic_splits \
    --checkpoint output/mimic_cxr/internimage_b/internimage_b_mimic_cxr_224/ckpt_epoch_best.pth \
    --split test \
    --batch-size 128 \
    --output output/mimic_cxr/internimage_b \
    --opts AMP_OPT_LEVEL O1

echo "=========================================="
echo "Evaluation finished at: $(date)"
echo "=========================================="
