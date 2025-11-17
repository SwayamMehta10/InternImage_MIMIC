# Quick Start Guide: InternImage on MIMIC-CXR

## Prerequisites
- Access to ASU Sol Supercomputer
- MIMIC-CXR dataset at `/scratch/aalla4/shared_folder/MIMIC/files`
- InternImage-B pretrained weights (ImageNet-1K)

## 5-Minute Setup

### 1. Clone and Navigate
```bash
cd d:/GitHub/InternImage_MIMIC/classification
```

### 2. Run Setup Script
```bash
bash setup_mimic_env.sh
```
This will:
- Create conda environment
- Install dependencies
- Build DCNv3 operators
- Create directories
- Update SLURM scripts

### 3. Download Pretrained Weights
```bash
cd pretrained
wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth
cd ..
```

### 4. Submit Training Job
```bash
sbatch train_mimic.sbatch
```

### 5. Monitor Training
```bash
# Check job status
squeue -u your_asurite

# View live log
tail -f logs/train_<job_id>.out
```

## After Training Completes

### Evaluate on Test Set
```bash
# Update checkpoint path in eval_mimic.sbatch
vim eval_mimic.sbatch  # Set CHECKPOINT_PATH to best model

# Submit evaluation
sbatch eval_mimic.sbatch

# View results
cat logs/eval_<job_id>.out
```

## Expected Results Location

```
classification/
├── output/mimic_cxr/internimage_b/run1/
│   └── best_ckpt.pth  ← Best model (highest val AUC-ROC)
└── logs/
    ├── train_<job_id>.out  ← Training logs
    └── eval_<job_id>.out   ← Evaluation results
```

## Key Metrics to Check

**During Training** (in logs/train_*.out):
- Train loss (should decrease)
- Val AUC-ROC (should increase, target: 0.75-0.85)
- Max AUC-ROC (best so far)

**After Evaluation** (in logs/eval_*.out):
- Mean AUC-ROC (macro-averaged over 14 diseases)
- Per-class AUC-ROC (for each of 14 diseases)

## Common Issues & Quick Fixes

### Issue: Job Exceeds 8-Hour Limit
**Solution**: Resume from checkpoint
```bash
# Edit train_mimic.sbatch, uncomment resume section
# Update: --resume output/mimic_cxr/internimage_b/run1/ckpt_epoch_X.pth
sbatch train_mimic.sbatch
```

### Issue: Out of GPU Memory
**Solution**: Reduce batch size
```bash
# Edit configs/internimage_b_mimic_cxr_224.yaml
# Change: BATCH_SIZE: 32 → BATCH_SIZE: 16
sbatch train_mimic.sbatch
```

### Issue: CSV File Not Found
**Solution**: Check dataset path and CSV naming
```bash
ls /scratch/aalla4/shared_folder/MIMIC/files/*.csv
# Update possible_names in dataset/mimic_cxr.py if needed
```

## File Overview

| File | Purpose |
|------|---------|
| `configs/internimage_b_mimic_cxr_224.yaml` | Model & training configuration |
| `dataset/mimic_cxr.py` | Dataset loader for MIMIC-CXR |
| `utils_multilabel.py` | AUC-ROC calculation |
| `train_mimic.sbatch` | SLURM training script |
| `eval_mimic.sbatch` | SLURM evaluation script |
| `README_MIMIC.md` | Detailed documentation |

## Customization

### Change Learning Rate
Edit `configs/internimage_b_mimic_cxr_224.yaml`:
```yaml
TRAIN:
  BASE_LR: 1e-4  # Change this (e.g., 5e-5, 2e-4)
```

### Change Number of Epochs
Edit `configs/internimage_b_mimic_cxr_224.yaml`:
```yaml
TRAIN:
  EPOCHS: 50  # Change this (e.g., 100)
```

### Use Smaller Model (InternImage-S)
1. Download InternImage-S weights
2. Create new config based on `configs/internimage_s_1k_224.yaml`
3. Set `NUM_CLASSES: 14` and adjust augmentations

## Training Time Estimates

- **Per Epoch**: ~4-5 minutes (depends on dataset size)
- **50 Epochs**: ~3-4 hours
- **100 Epochs**: ~6-8 hours (requires 2 job submissions due to 8-hour limit)

## Success Indicators

✓ Training loss decreases steadily  
✓ Validation AUC-ROC increases  
✓ Final mean AUC-ROC > 0.75  
✓ No NaN losses or errors  
✓ Checkpoints saved successfully  

## Next Steps After Successful Training

1. Analyze per-class AUC-ROC scores
2. Identify best/worst performing diseases
3. Consider extended training for improvement
4. Try different hyperparameters
5. Experiment with larger image sizes (384x384)

## Getting Help

- **Setup Issues**: See `README_MIMIC.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Dataset Format**: Check `dataset/mimic_cxr.py` comments
- **Sol-Specific**: ASU Research Computing support

---

**Ready to Start?**
```bash
bash setup_mimic_env.sh && sbatch train_mimic.sbatch
```
