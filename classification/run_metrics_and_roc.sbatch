#!/bin/bash
#SBATCH -J metrics_roc
#SBATCH -p public
#SBATCH -q class
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/metrics_roc_%j.out
#SBATCH --error=logs/metrics_roc_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smehta90@asu.edu

# Generate comprehensive metrics and ROC curves for MIMIC-CXR test set results

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

# Activate environment
source activate INTERNIMAGE_ENV

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Calculating comprehensive metrics..."
echo "=========================================="

python calculate_metrics.py \
    --cfg configs/internimage_b_mimic_cxr_224.yaml \
    --checkpoint output/mimic_cxr/internimage_b/internimage_b_mimic_cxr_224/ckpt_epoch_best.pth \
    --data-path /scratch/smehta90/mimic_splits \
    --split test \
    --batch-size 128 \
    --output output/mimic_cxr/internimage_b/metrics

echo ""
echo "=========================================="
echo "Generating ROC curve visualizations..."
echo "=========================================="

python plot_roc_curves.py \
    --cfg configs/internimage_b_mimic_cxr_224.yaml \
    --checkpoint output/mimic_cxr/internimage_b/internimage_b_mimic_cxr_224/ckpt_epoch_best.pth \
    --data-path /scratch/smehta90/mimic_splits \
    --split test \
    --batch-size 128 \
    --output output/mimic_cxr/internimage_b/roc_curves

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

echo ""
echo "Results saved to:"
echo "  Metrics: output/mimic_cxr/internimage_b/metrics/"
echo "  ROC Curves: output/mimic_cxr/internimage_b/roc_curves/"
