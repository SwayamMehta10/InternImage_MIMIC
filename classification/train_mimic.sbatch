#!/bin/bash
#SBATCH -J internimage_mimic
#SBATCH -p public
#SBATCH -q class
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

# InternImage training script for MIMIC-CXR on ASU Sol Supercomputer
# This script trains InternImage-B on MIMIC-CXR disease classification

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules on ASU Sol
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

# Activate conda environment
# Make sure to create this environment beforehand with required packages
source activate INTERNIMAGE_ENV

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Set distributed training environment variables for single GPU
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Print environment info
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "GPU info:"
nvidia-smi

echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Training command
# Adjust parameters as needed based on your setup
python main.py \
    --cfg configs/internimage_b_mimic_cxr_224.yaml \
    --data-path /scratch/aalla4/shared_folder/MIMIC/files \
    --batch-size 32 \
    --pretrained pretrained/internimage_b_1k_224.pth \
    --output output/mimic_cxr/internimage_b \
    --tag run1 \
    --local-rank 0 \
    --opts AMP_OPT_LEVEL O1 \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

# To resume from a checkpoint, uncomment and modify:
# python main.py \
#     --cfg configs/internimage_b_mimic_cxr_224.yaml \
#     --data-path /scratch/aalla4/shared_folder/MIMIC/files \
#     --batch-size 32 \
#     --resume output/mimic_cxr/internimage_b/run1/ckpt_epoch_X.pth \
#     --output output/mimic_cxr/internimage_b \
#     --tag run1 \
#     --local-rank 0 \
#     --opts AMP_OPT_LEVEL O1 \
#     2>&1 | tee -a logs/train_resume_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Print job statistics
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
