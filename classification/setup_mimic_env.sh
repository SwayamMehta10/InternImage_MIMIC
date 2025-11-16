#!/bin/bash
# Setup script for InternImage MIMIC-CXR training on ASU Sol
# Run this script after cloning the repository

set -e  # Exit on error

echo "=========================================="
echo "InternImage MIMIC-CXR Setup Script"
echo "=========================================="

# Check if running on Sol
if [[ ! $(hostname) =~ "sol" ]]; then
    echo "Warning: This script is designed for ASU Sol supercomputer"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Navigate to classification directory
cd classification

echo ""
echo "Step 1: Loading modules..."
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest


echo ""
echo "Step 2: Creating conda environment..."
mamba create -n INTERNIMAGE_ENV -c conda-forge python=3.12
source activate INTERNIMAGE_ENV

echo ""
echo "Step 3: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "Step 4: Installing other dependencies..."
pip install -r requirements_mimic.txt

echo ""
echo "Step 5: Building DCNv3 operators..."
module load gcc-12.1.0-gcc-11.2.0
gcc --version  # Verify GCC 12.1.0 is loaded
cd ops_dcnv3
if [ -f "make.sh" ]; then
    bash make.sh
    echo "DCNv3 operators built successfully"
else
    echo "Error: make.sh not found in ops_dcnv3"
    exit 1
fi
cd ..

echo ""
echo "Step 6: Creating necessary directories..."
mkdir -p logs
mkdir -p output/mimic_cxr/internimage_b
mkdir -p pretrained
mkdir -p results/mimic_cxr

echo ""
echo "Step 7: Checking dataset path..."
DATASET_PATH="/scratch/aalla4/shared_folder/MIMIC/files"
if [ -d "$DATASET_PATH" ]; then
    echo "✓ Dataset path exists: $DATASET_PATH"
    # Try to find CSV files
    echo "Looking for CSV files..."
    find $DATASET_PATH -maxdepth 1 -name "*.csv" -type f | head -5
else
    echo "⚠ Warning: Dataset path not found: $DATASET_PATH"
    echo "Please verify the dataset location."
fi

echo ""
echo "Step 8: Checking for pretrained weights..."
PRETRAINED_PATH="pretrained/internimage_b_1k_224.pth"
if [ -f "$PRETRAINED_PATH" ]; then
    echo "✓ Pretrained weights found: $PRETRAINED_PATH"
else
    echo "⚠ Pretrained weights not found"
    echo "Please download InternImage-B weights to: $PRETRAINED_PATH"
    echo "URL: https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth"
    echo ""
    read -p "Download now? (requires internet access) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wget -O $PRETRAINED_PATH https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify dataset location and CSV files"
echo "2. Download pretrained weights if not done"
echo "3. Review configuration: configs/internimage_b_mimic_cxr_224.yaml"
echo "4. Submit training job: sbatch train_mimic.sbatch"
echo ""
echo "Environment: INTERNIMAGE_ENV"
echo "To activate: source activate INTERNIMAGE_ENV"
echo ""
echo "For more information, see README_MIMIC.md"
echo "=========================================="
