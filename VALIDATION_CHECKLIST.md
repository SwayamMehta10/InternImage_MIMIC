# Pre-Training Validation Checklist

Use this checklist to verify your setup before submitting training jobs.

## ☐ Environment Setup

### Conda Environment
- [ ] Conda environment created: `conda env list | grep internimage_env`
- [ ] PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] All packages installed: `pip list | grep -E "timm|yacs|sklearn|pandas"`

### DCNv3 Operators
- [ ] DCNv3 compiled: `ls ops_dcnv3/build/` should show compiled files
- [ ] DCNv3 import test:
  ```bash
  python -c "from ops_dcnv3.modules import DCNv3; print('DCNv3 OK')"
  ```

## ☐ Dataset Verification

### Dataset Location
- [ ] Dataset path exists:
  ```bash
  ls -la /scratch/aalla4/shared_folder/MIMIC/files/
  ```

### CSV Files
- [ ] Find CSV files:
  ```bash
  find /scratch/aalla4/shared_folder/MIMIC/files -name "*.csv" -type f
  ```
- [ ] CSV contains required columns (check one):
  ```bash
  head -n 1 /scratch/aalla4/shared_folder/MIMIC/files/<your_csv_file>.csv
  ```
  Expected: path column + 14 disease label columns

### Image Files
- [ ] Images accessible:
  ```bash
  find /scratch/aalla4/shared_folder/MIMIC/files -name "*.jpg" -o -name "*.png" -o -name "*.dcm" | head -5
  ```

## ☐ Pretrained Weights

- [ ] Weights downloaded:
  ```bash
  ls -lh classification/pretrained/internimage_b_1k_224.pth
  ```
- [ ] File size reasonable: ~400-500 MB
- [ ] Can load weights:
  ```bash
  python -c "import torch; torch.load('classification/pretrained/internimage_b_1k_224.pth'); print('Weights OK')"
  ```

## ☐ Configuration Files

### Main Config
- [ ] Config file exists: `classification/configs/internimage_b_mimic_cxr_224.yaml`
- [ ] Key settings verified:
  ```bash
  grep -A 2 "DATASET:" classification/configs/internimage_b_mimic_cxr_224.yaml
  grep -A 2 "NUM_CLASSES:" classification/configs/internimage_b_mimic_cxr_224.yaml
  grep -A 2 "BASE_LR:" classification/configs/internimage_b_mimic_cxr_224.yaml
  ```
- [ ] Expected values:
  - DATASET: 'mimic_cxr'
  - NUM_CLASSES: 14
  - BASE_LR: 1e-4

## ☐ SLURM Scripts

### Training Script
- [ ] Script exists: `classification/train_mimic.sbatch`
- [ ] Email updated: `grep "mail-user" classification/train_mimic.sbatch`
- [ ] Module versions correct:
  ```bash
  grep "module load" classification/train_mimic.sbatch
  ```
- [ ] Executable: `chmod +x classification/train_mimic.sbatch`

### Evaluation Script
- [ ] Script exists: `classification/eval_mimic.sbatch`
- [ ] Email updated: `grep "mail-user" classification/eval_mimic.sbatch`
- [ ] Executable: `chmod +x classification/eval_mimic.sbatch`

## ☐ Directory Structure

- [ ] Required directories exist:
  ```bash
  ls -d classification/{logs,output,pretrained,results}
  ```
- [ ] Sufficient disk space:
  ```bash
  df -h $PWD
  ```
  (Need ~50GB for checkpoints and logs)

## ☐ Code Validation

### Import Test
- [ ] All custom modules import correctly:
  ```bash
  cd classification
  python -c "
  from dataset.mimic_cxr import MIMICCXRDataset
  from utils_multilabel import validate_multilabel
  print('All imports OK')
  "
  ```

### Dataset Test
- [ ] Dataset can be instantiated:
  ```bash
  cd classification
  python -c "
  from dataset.mimic_cxr import MIMICCXRDataset
  import torchvision.transforms as transforms
  
  transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
  ])
  
  try:
      dataset = MIMICCXRDataset(
          data_root='/scratch/aalla4/shared_folder/MIMIC/files',
          split='train',
          transform=transform
      )
      print(f'Dataset OK: {len(dataset)} samples')
      img, label = dataset[0]
      print(f'Image shape: {img.shape}, Label shape: {label.shape}')
  except Exception as e:
      print(f'Dataset Error: {e}')
  "
  ```

## ☐ Permissions & Access

- [ ] Can submit SLURM jobs: `sbatch --test-only classification/train_mimic.sbatch`
- [ ] Can access GPU partition: `sinfo -p public | grep gpu`
- [ ] Enough quota in home directory: `quota -s`

## ☐ Module Availability (on Sol)

- [ ] Check available modules:
  ```bash
  module avail anaconda
  module avail cuda
  ```
- [ ] Load test:
  ```bash
  module purge
  module load anaconda3/2023.03
  module load cuda/11.8
  which python
  which nvcc
  ```

## ☐ Final Pre-Flight Check

### Quick Dry Run (Optional but Recommended)
Test training for 1 iteration:
```bash
cd classification
python main.py \
  --cfg configs/internimage_b_mimic_cxr_224.yaml \
  --data-path /scratch/aalla4/shared_folder/MIMIC/files \
  --batch-size 4 \
  --opts TRAIN.EPOCHS 1 PRINT_FREQ 1
```

Expected: Should load data, start training, print first iteration, then you can Ctrl+C

### Checklist Summary
- [ ] Environment: Conda + PyTorch + Dependencies ✓
- [ ] Dataset: Path + CSVs + Images accessible ✓
- [ ] Weights: Pretrained model downloaded ✓
- [ ] Config: Settings verified ✓
- [ ] Scripts: SLURM scripts ready ✓
- [ ] Code: Imports and dataset working ✓
- [ ] Permissions: Can submit jobs ✓

## 🚀 Ready to Launch!

If all items are checked, you're ready to submit:
```bash
cd classification
sbatch train_mimic.sbatch
```

Monitor with:
```bash
squeue -u $USER
tail -f logs/train_*.out
```

## Common Issues Found During Validation

### Issue: DCNv3 import fails
**Fix**: Rebuild operators
```bash
cd classification/ops_dcnv3
bash make.sh
```

### Issue: Dataset CSV not found
**Fix**: Check CSV naming in `dataset/mimic_cxr.py` line 62-67
```python
possible_names = [
    f'mimic-cxr-2.0.0-chexpert-{split}.csv',
    f'mimic_cxr_{split}.csv',
    f'{split}.csv',
    'mimic-cxr-2.0.0-chexpert.csv',
]
```
Add your CSV naming pattern to this list.

### Issue: Module load fails
**Fix**: Check exact module names on Sol
```bash
module spider anaconda
module spider cuda
```
Update module names in SLURM scripts accordingly.

### Issue: Permission denied on dataset path
**Fix**: Verify group membership and path permissions
```bash
ls -la /scratch/aalla4/shared_folder/MIMIC/
groups  # Check your groups
```

### Issue: Insufficient disk quota
**Fix**: Clean up old files or request quota increase
```bash
du -sh ~/.*  # Find large hidden directories
du -sh *     # Find large directories
```

## Support

If validation fails at any step:
1. Check the error message carefully
2. Consult `README_MIMIC.md` troubleshooting section
3. Review `IMPLEMENTATION_SUMMARY.md` for technical details
4. Contact ASU Research Computing for Sol-specific issues

---

**Note**: It's highly recommended to complete this checklist before your first training run to avoid wasted compute time debugging during training.
