# MIMIC-CXR Disease Classification with InternImage

This directory contains the implementation for training InternImage on the MIMIC-CXR dataset for multi-label disease classification.

## Overview

- **Task**: Multi-label disease classification (14 diseases)
- **Model**: InternImage-B (97M parameters)
- **Dataset**: MIMIC-CXR chest X-ray images
- **Metrics**: Train/Val Loss, AUC-ROC (per-class and macro-averaged)
- **Platform**: ASU Sol Supercomputer with A100 GPU

## Dataset Structure

The MIMIC-CXR dataset should be located at `/scratch/aalla4/shared_folder/MIMIC/files/` with the following structure:

```
/scratch/aalla4/shared_folder/MIMIC/files/
├── mimic-cxr-2.0.0-chexpert-train.csv  (or similar naming)
├── mimic-cxr-2.0.0-chexpert-val.csv
├── mimic-cxr-2.0.0-chexpert-test.csv
└── images/  (or subdirectories with image files)
```

### Disease Labels (14 classes)
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Enlarged Cardiomediastinum
6. Fracture
7. Lung Lesion
8. Lung Opacity
9. No Finding
10. Pleural Effusion
11. Pleural Other
12. Pneumonia
13. Pneumothorax
14. Support Devices

## Setup Instructions

### 1. Environment Setup on ASU Sol

```bash
# SSH to ASU Sol
ssh your_asurite@sol.asu.edu

# Load conda module
module load anaconda3/2023.03

# Create conda environment
conda create -n internimage_env python=3.9 -y
conda activate internimage_env

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install required packages
pip install -r requirements_mimic.txt

# Build DCNv3 operators
cd ops_dcnv3
bash make.sh
cd ..
```

### 2. Download Pretrained Weights

Download InternImage-B ImageNet-1K pretrained weights:

```bash
mkdir -p pretrained
cd pretrained

# Download from HuggingFace or official source
# Place the file as: internimage_b_1k_224.pth
# URL: https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth

cd ..
```

### 3. Prepare SLURM Scripts

Update the email address in the SLURM scripts:
- `train_mimic.sbatch`: Line with `--mail-user=your_email@asu.edu`
- `eval_mimic_sol.sh`: Line with `--mail-user=your_email@asu.edu`

## Training

### Submit Training Job

```bash
cd classification
sbatch train_mimic.sbatch
```

### Monitor Training

```bash
# Check job status
squeue -u your_asurite

# View training log
tail -f logs/train_<job_id>.out

# Or view specific log file
tail -f logs/train_<timestamp>.log
```

### Resume Training from Checkpoint

If training is interrupted (8-hour walltime limit), update `train_mimic.sbatch`:

```bash
# Uncomment the resume section in train_mimic.sbatch
# Update the checkpoint path to the last saved epoch
--resume output/mimic_cxr/internimage_b/run1/ckpt_epoch_X.pth
```

Then resubmit:
```bash
sbatch train_mimic.sbatch
```

## Evaluation

### Test Set Evaluation

After training completes, evaluate on the test set:

```bash
# Update checkpoint path in eval_mimic_sol.sh if needed
sbatch eval_mimic_sol.sh
```

### View Results

```bash
# Check evaluation log
cat logs/eval_<job_id>.out

# Results include:
# - Mean AUC-ROC across all 14 diseases
# - Per-class AUC-ROC scores
# - Average loss
```

## Key Files

### Core Implementation
- `dataset/mimic_cxr.py`: MIMIC-CXR dataset loader with multi-label support
- `utils_multilabel.py`: Multi-label validation and AUC-ROC calculation
- `main.py`: Training script (modified for multi-label)
- `configs/internimage_b_mimic_cxr_224.yaml`: Configuration file

### SLURM Scripts
- `train_mimic.sbatch`: Training job submission script
- `eval_mimic_sol.sh`: Evaluation job submission script

## Configuration Details

Key settings in `configs/internimage_b_mimic_cxr_224.yaml`:

```yaml
DATA:
  DATASET: 'mimic_cxr'
  BATCH_SIZE: 32
  IMG_SIZE: 224

MODEL:
  NUM_CLASSES: 14  # 14 diseases
  
TRAIN:
  EPOCHS: 50
  BASE_LR: 1e-4  # Lower LR for medical imaging
  WARMUP_EPOCHS: 5
  
AUG:
  COLOR_JITTER: 0.0  # Disabled for grayscale
  MIXUP: 0.0  # Disabled for multi-label
  CUTMIX: 0.0  # Disabled for multi-label
```

## Expected Performance

Training details:
- **Epochs**: 50 (can be extended)
- **Batch Size**: 32 per GPU
- **Training Time**: ~3-4 hours per 50 epochs on A100
- **GPU Memory**: ~20-25GB
- **Checkpoints**: Saved every 5 epochs + best model

Expected metrics:
- **Mean AUC-ROC**: 0.75-0.85 (depends on data quality and training)
- **Per-class AUC-ROC**: Varies by disease prevalence

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `BATCH_SIZE` in config (try 16 or 24)
   - Enable gradient checkpointing: `USE_CHECKPOINT: True`
   - Increase `ACCUMULATION_STEPS` to simulate larger batch size

2. **CSV File Not Found**
   - Check CSV naming convention in dataset loader
   - Verify path in `DATA_PATH` config
   - Update `possible_names` list in `mimic_cxr.py` if needed

3. **Module Not Found**
   - Ensure all packages from `requirements_mimic.txt` are installed
   - Rebuild DCNv3 operators: `cd ops_dcnv3 && bash make.sh`

4. **Job Time Limit**
   - Training saves checkpoints every 5 epochs automatically
   - Resume using the `--resume` flag with latest checkpoint

5. **CUDA/Module Issues**
   - Check available modules: `module avail cuda`
   - Load correct CUDA version matching PyTorch installation

## Output Structure

```
classification/
├── output/mimic_cxr/internimage_b/run1/
│   ├── ckpt_epoch_5.pth
│   ├── ckpt_epoch_10.pth
│   ├── ...
│   ├── best_ckpt.pth  # Best model based on validation AUC-ROC
│   └── config.json
├── logs/
│   ├── train_<job_id>.out
│   ├── train_<timestamp>.log
│   └── eval_<job_id>.out
└── results/mimic_cxr/
    └── evaluation results
```

## Contact & Support

For issues specific to:
- **InternImage model**: Refer to original InternImage repository
- **MIMIC-CXR dataset**: Check MIMIC-CXR documentation
- **ASU Sol**: Contact ASU Research Computing support

## Citation

If you use this implementation, please cite:

```bibtex
@article{internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and Wang, Xiaogang and Qiao, Yu},
  journal={CVPR},
  year={2023}
}

@article{johnson2019mimic,
  title={MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs},
  author={Johnson, Alistair EW and Pollard, Tom J and Berkowitz, Seth and Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and Mark, Roger G and Horng, Steven},
  journal={arXiv preprint arXiv:1901.07042},
  year={2019}
}
```

## License

This code follows the MIT License from the original InternImage repository.
