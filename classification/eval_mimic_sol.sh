#!/bin/bash
#SBATCH --job-name=internimage_eval
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@asu.edu

# InternImage evaluation script for MIMIC-CXR on ASU Sol Supercomputer
# This script evaluates trained InternImage-B on MIMIC-CXR test set

echo "=========================================="
echo "Evaluation job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules on ASU Sol
module purge
module load anaconda3/2023.03  # Adjust version as needed
module load cuda/11.8  # Adjust CUDA version as needed

# Activate conda environment
source activate internimage_env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p logs
mkdir -p results/mimic_cxr

# Navigate to classification directory
cd classification

# Print environment info
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "=========================================="
echo "Starting evaluation on test set..."
echo "=========================================="

# Set checkpoint path - UPDATE THIS to your best model checkpoint
CHECKPOINT_PATH="output/mimic_cxr/internimage_b/run1/best_ckpt.pth"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please update the CHECKPOINT_PATH variable in this script"
    exit 1
fi

# Evaluation command
python main.py \
    --cfg configs/internimage_b_mimic_cxr_224.yaml \
    --data-path /scratch/aalla4/shared_folder/MIMIC/files \
    --batch-size 64 \
    --eval \
    --resume $CHECKPOINT_PATH \
    --output results/mimic_cxr \
    --tag evaluation \
    --opts AMP_OPT_LEVEL O1 \
    2>&1 | tee logs/eval_$(date +%Y%m%d_%H%M%S).log

# Alternative: Evaluate with EMA model if available
# Uncomment to use EMA checkpoint instead
# EMA_CHECKPOINT_PATH="output/mimic_cxr/internimage_b/run1/ema_best_ckpt.pth"
# python main.py \
#     --cfg configs/internimage_b_mimic_cxr_224.yaml \
#     --data-path /scratch/aalla4/shared_folder/MIMIC/files \
#     --batch-size 64 \
#     --eval \
#     --resume $EMA_CHECKPOINT_PATH \
#     --output results/mimic_cxr \
#     --tag evaluation_ema \
#     --opts AMP_OPT_LEVEL O1 \
#     2>&1 | tee logs/eval_ema_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "Evaluation finished at: $(date)"
echo "=========================================="

# Print job statistics
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

# Print location of results
echo "=========================================="
echo "Results saved to: results/mimic_cxr/"
echo "Logs saved to: logs/eval_*.log"
echo "=========================================="
