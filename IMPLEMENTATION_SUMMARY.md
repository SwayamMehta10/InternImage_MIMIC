# InternImage MIMIC-CXR Training Implementation Summary

## Overview

This implementation adapts the InternImage classification codebase for multi-label disease classification on the MIMIC-CXR chest X-ray dataset. The system is configured for ASU Sol Supercomputer with A100 GPU support.

## Key Modifications Made

### 1. Dataset Loader (NEW: `classification/dataset/mimic_cxr.py`)

**Purpose**: Load MIMIC-CXR chest X-rays with multi-label annotations

**Features**:
- Loads images from CSV files with 14 disease labels
- Handles both DICOM and standard image formats
- Converts grayscale X-rays to RGB (3-channel) for ImageNet-pretrained models
- Manages missing/uncertain labels (NaN, -1) by treating as negative
- Returns float tensors compatible with BCEWithLogitsLoss
- Supports official train/val/test splits

**Key Methods**:
- `__init__()`: Load CSV, extract paths and labels
- `__getitem__()`: Load image, apply transforms, return (image, label) tuple
- `_load_dicom()`: DICOM file handling with VOI LUT windowing

### 2. Multi-Label Utilities (NEW: `classification/utils_multilabel.py`)

**Purpose**: Validation and metric calculation for multi-label classification

**Key Functions**:
- `validate_multilabel()`: Validation loop with BCEWithLogitsLoss
  - Collects predictions and targets across all batches
  - Applies sigmoid to convert logits to probabilities
  - Computes per-class AUC-ROC using scikit-learn
  - Handles classes with single label value gracefully
  - Returns mean AUC, loss, and per-class AUC scores

- `evaluate_test_set()`: Detailed test set evaluation
  - Wrapper around validate_multilabel for final evaluation
  - Returns comprehensive metrics dictionary
  - Logs results in formatted output

### 3. Training Script Modifications (`classification/main.py`)

**Changes Made**:

1. **Import multi-label utilities**:
   ```python
   from utils_multilabel import validate_multilabel, evaluate_test_set
   ```

2. **Loss function selection**:
   ```python
   if config.DATA.DATASET == 'mimic_cxr':
       criterion = torch.nn.BCEWithLogitsLoss()
   ```

3. **Validation metric branching**:
   - Single-label: Uses accuracy (top-1, top-5)
   - Multi-label (MIMIC-CXR): Uses AUC-ROC
   - Applied in: pretrained loading, checkpoint resume, training loop, EMA evaluation

4. **Best model tracking**:
   - For MIMIC-CXR: Track max AUC-ROC instead of max accuracy
   - Save best checkpoint based on validation AUC-ROC

### 4. Dataset Integration (`classification/dataset/`)

**Updated Files**:
- `__init__.py`: Import MIMICCXRDataset
- `build.py`: Add MIMIC-CXR case in build_dataset()
  ```python
  elif config.DATA.DATASET == 'mimic_cxr':
      dataset = MIMICCXRDataset(root, split=prefix, transform=transform)
      nb_classes = 14
  ```

### 5. Configuration (NEW: `classification/configs/internimage_b_mimic_cxr_224.yaml`)

**Key Settings**:

**Data**:
- `DATASET: 'mimic_cxr'`
- `DATA_PATH: '/scratch/aalla4/shared_folder/MIMIC/files'`
- `BATCH_SIZE: 32`
- `IMG_SIZE: 224`

**Model**:
- `NUM_CLASSES: 14` (14 diseases)
- `DROP_PATH_RATE: 0.5` (InternImage-B)
- InternImage-B architecture: depths=[4,4,21,4], channels=112

**Training**:
- `EPOCHS: 50`
- `BASE_LR: 1e-4` (lower for medical imaging)
- `WARMUP_EPOCHS: 5`
- `EMA.ENABLE: True` (EMA decay: 0.9999)

**Augmentation**:
- `COLOR_JITTER: 0.0` (disabled for grayscale)
- `MIXUP: 0.0` (disabled for multi-label)
- `CUTMIX: 0.0` (disabled for multi-label)
- `REPROB: 0.1` (reduced random erasing)

### 6. SLURM Scripts

**Training Script** (`classification/train_mimic.sbatch`):
- Job configuration: 1 A100 GPU, 8 CPUs, 64GB RAM, 8-hour walltime
- Partition: public, QoS: class
- Loads modules: anaconda3, cuda/11.8
- Activates conda environment
- Runs training with automatic logging
- Includes commented resume section for checkpointing
- Prints job statistics at end

**Evaluation Script** (`classification/eval_mimic.sbatch`):
- Job configuration: 1 A100 GPU, 4 CPUs, 32GB RAM, 1-hour walltime
- Evaluates best checkpoint on test set
- Supports EMA model evaluation (commented)
- Outputs detailed per-class AUC-ROC scores

### 7. Documentation

**README_MIMIC.md**:
- Complete setup instructions
- Dataset structure requirements
- Training and evaluation workflows
- Troubleshooting guide
- Expected performance metrics

**requirements_mimic.txt**:
- All Python dependencies
- Includes pydicom for DICOM support
- scikit-learn for AUC-ROC metrics

**setup_mimic_env.sh**:
- Automated environment setup script
- Builds DCNv3 operators
- Creates necessary directories
- Downloads pretrained weights
- Updates SLURM scripts with user email

## Training Pipeline

### Phase 1: Initial Training (Epochs 1-50)
1. Load pretrained ImageNet-1K weights
2. Train full model with lower learning rate (1e-4)
3. Use cosine annealing schedule with 5-epoch warmup
4. Save checkpoints every 5 epochs
5. Track best model by validation AUC-ROC

### Phase 2: Extended Training (Optional)
- Resume from best checkpoint
- Train additional epochs if needed
- Can adjust learning rate for fine-tuning

### Checkpointing Strategy
- Auto-save every 5 epochs
- Best model saved separately
- EMA model tracked and saved
- Enables resume after 8-hour walltime limit

## Metrics Tracking

### Training Metrics (per epoch)
- Train loss (BCEWithLogitsLoss)
- Learning rate
- Gradient norm
- Training time

### Validation Metrics (per epoch)
- Validation loss
- Mean AUC-ROC (macro-averaged over 14 classes)
- Per-class AUC-ROC scores
- Best AUC-ROC so far

### Test Metrics (final evaluation)
- Test loss
- Mean AUC-ROC
- Per-class AUC-ROC for all 14 diseases
- Detailed results logged and saved

## Expected Outputs

### Directory Structure
```
classification/
├── output/mimic_cxr/internimage_b/run1/
│   ├── ckpt_epoch_5.pth
│   ├── ckpt_epoch_10.pth
│   ├── ...
│   ├── ckpt_epoch_50.pth
│   ├── best_ckpt.pth
│   ├── ema_best_ckpt.pth (if EMA enabled)
│   └── config.json
├── logs/
│   ├── train_<job_id>.out
│   ├── train_<timestamp>.log
│   └── eval_<job_id>.out
└── results/mimic_cxr/
    └── evaluation_<timestamp>/
```

### Checkpoint Contents
Each checkpoint contains:
- `model`: Model state dict
- `optimizer`: Optimizer state
- `lr_scheduler`: Learning rate scheduler state
- `epoch`: Current epoch number
- `max_accuracy`: Best validation AUC-ROC so far
- `config`: Full configuration
- `amp`: AMP scaler state (if using mixed precision)
- `model_ema`: EMA model state (if enabled)

## Key Design Decisions

### 1. Multi-Label Loss
**Choice**: BCEWithLogitsLoss
**Reason**: 
- Combines sigmoid and BCE for numerical stability
- Standard for multi-label classification
- No need for manual sigmoid in training loop

### 2. Missing Label Handling
**Choice**: Treat NaN and -1 (uncertain) as negative (0)
**Reason**:
- Simplest approach, no class imbalance handling required
- Follows user requirement for official splits without modifications
- Alternative approaches (masking, exclusion) can be added if needed

### 3. Grayscale to RGB Conversion
**Choice**: Repeat grayscale channel 3 times
**Reason**:
- Maintains compatibility with ImageNet-pretrained weights
- No architecture changes needed
- Proven effective in medical imaging transfer learning

### 4. Learning Rate
**Choice**: 1e-4 (10x lower than ImageNet)
**Reason**:
- Medical images require careful fine-tuning
- Prevents catastrophic forgetting of pretrained features
- Can be adjusted based on validation performance

### 5. Augmentation Strategy
**Choice**: Minimal augmentation (no color jitter, mixup, cutmix)
**Reason**:
- Grayscale medical images
- Multi-label classification (mixup incompatible)
- Preserve clinical features
- Random erasing kept at low probability (0.1)

## Files Created/Modified

### New Files (8)
1. `classification/dataset/mimic_cxr.py` - Dataset loader
2. `classification/utils_multilabel.py` - Multi-label utilities
3. `classification/configs/internimage_b_mimic_cxr_224.yaml` - Configuration
4. `classification/train_mimic.sbatch` - Training SLURM script
5. `classification/eval_mimic.sbatch` - Evaluation SLURM script
6. `classification/README_MIMIC.md` - Documentation
7. `classification/requirements_mimic.txt` - Dependencies
8. `classification/setup_mimic_env.sh` - Setup script

### Modified Files (3)
1. `classification/dataset/__init__.py` - Import MIMICCXRDataset
2. `classification/dataset/build.py` - Add MIMIC-CXR support
3. `classification/main.py` - Multi-label training logic

## Usage Workflow

### Setup (One-time)
```bash
cd classification
bash setup_mimic_env.sh
# Follow prompts for email, environment name, etc.
```

### Training
```bash
sbatch train_mimic.sbatch
squeue -u your_asurite  # Check status
tail -f logs/train_<job_id>.out  # Monitor progress
```

### Resume (if needed)
```bash
# Edit train_mimic.sbatch to uncomment resume section
# Update checkpoint path
sbatch train_mimic.sbatch
```

### Evaluation
```bash
# Edit eval_mimic.sbatch to set checkpoint path
sbatch eval_mimic.sbatch
cat logs/eval_<job_id>.out  # View results
```

## Next Steps for User

1. **Before First Run**:
   - Verify MIMIC-CXR dataset location and CSV files
   - Download pretrained InternImage-B weights
   - Update email in SLURM scripts
   - Review and adjust config if needed

2. **During Training**:
   - Monitor logs for convergence
   - Check validation AUC-ROC trends
   - Be prepared to resume after 8-hour limit

3. **After Training**:
   - Evaluate best checkpoint on test set
   - Analyze per-class AUC-ROC scores
   - Consider extended training if needed

4. **Potential Improvements**:
   - Try larger image size (384x384) for better performance
   - Experiment with class weighting for imbalanced classes
   - Test different learning rates and warmup schedules
   - Compare with InternImage-S (smaller) or InternImage-L (larger)

## Technical Notes

### Memory Management
- Batch size 32 fits comfortably on A100 40GB
- Can increase to 64 if memory allows
- Gradient accumulation available if needed
- Gradient checkpointing can reduce memory further

### Multi-GPU Support
- Current setup: Single GPU (1 A100)
- Code supports DDP for multi-GPU
- Would require SLURM script modification

### Data Loading
- 8 workers for data loading
- Images loaded on-the-fly (not cached in memory)
- Can enable `IMG_ON_MEMORY: True` if dataset fits in RAM

### Reproducibility
- Random seed: 42 (set in config)
- Deterministic mode available via PyTorch settings
- Checkpoint-based resume ensures reproducible training

## Contact for Issues

- Dataset issues: Check MIMIC-CXR documentation
- Model issues: Refer to InternImage repository
- Sol-specific: ASU Research Computing support
- Implementation bugs: Check this summary and README

---

**Implementation Date**: November 2025
**Platform**: ASU Sol Supercomputer
**Model**: InternImage-B (97M parameters)
**Task**: MIMIC-CXR Multi-Label Disease Classification (14 classes)
