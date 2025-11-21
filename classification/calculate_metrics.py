"""
Calculate comprehensive metrics for MIMIC-CXR multi-label classification results.

Calculates:
- Precision, Recall, F1-Score per class (at threshold 0.5)
- Specificity, Sensitivity at threshold 0.5
- Average Precision (AP) scores
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@torch.no_grad()
def calculate_comprehensive_metrics(config, data_loader, model, output_dir='output/metrics'):
    """Calculate comprehensive metrics for multi-label classification."""
    
    model.eval()
    
    # Collect all predictions and targets
    all_outputs = []
    all_targets = []
    
    print("Collecting predictions...")
    for idx, (images, target) in enumerate(data_loader):
        if isinstance(images, list):
            images = [item.cuda(non_blocking=True) for item in images]
        else:
            images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        output = model(images)
        all_outputs.append(output.cpu())
        all_targets.append(target.cpu())
        
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(data_loader)} batches")
    
    # Concatenate all predictions
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Apply sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_outputs))
    
    # Disease labels
    disease_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
        'Pneumothorax', 'Support Devices'
    ]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics for each class
    results = []
    print("\nCalculating metrics per class...")
    
    for i, label in enumerate(disease_labels):
        y_true = all_targets[:, i]
        y_prob = all_probs[:, i]
        
        # Skip if only one class present
        if len(np.unique(y_true)) == 1:
            print(f"  Skipping {label}: only one class present")
            continue
        
        # AUC-ROC
        auc_roc = roc_auc_score(y_true, y_prob)
        
        # Average Precision (AP)
        ap = average_precision_score(y_true, y_prob)
        
        # Calculate metrics at default 0.5 threshold
        y_pred_05 = (y_prob >= 0.5).astype(int)
        
        # Calculate metrics at 0.5 threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_05 = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1_05 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        recall_05 = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Store results
        results.append({
            'Disease': label,
            'AUC-ROC': auc_roc,
            'AP': ap,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision_05,
            'NPV': npv,
            'F1': f1_05,
            'Recall': recall_05,
            'Positive_Samples': int(y_true.sum()),
            'Negative_Samples': int((1 - y_true).sum())
        })
        
        print(f"  {label}: AUC-ROC={auc_roc:.4f}, AP={ap:.4f}, F1={f1_05:.4f}")
    
    # Save detailed results
    import pandas as pd
    df_results = pd.DataFrame(results)
    
    # Calculate mean metrics
    mean_metrics = {
        'Disease': 'MEAN',
        'AUC-ROC': df_results['AUC-ROC'].mean(),
        'AP': df_results['AP'].mean(),
        'Sensitivity': df_results['Sensitivity'].mean(),
        'Specificity': df_results['Specificity'].mean(),
        'Precision': df_results['Precision'].mean(),
        'NPV': df_results['NPV'].mean(),
        'F1': df_results['F1'].mean(),
        'Recall': df_results['Recall'].mean(),
        'Positive_Samples': df_results['Positive_Samples'].sum(),
        'Negative_Samples': df_results['Negative_Samples'].sum()
    }
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_metrics])], ignore_index=True)
    
    # Save to CSV
    csv_path = Path(output_dir) / 'comprehensive_metrics.csv'
    df_results.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved comprehensive metrics to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY METRICS (threshold = 0.5)")
    print("="*80)
    print(f"Mean AUC-ROC:        {mean_metrics['AUC-ROC']:.4f}")
    print(f"Mean AP:             {mean_metrics['AP']:.4f}")
    print(f"Mean Sensitivity:    {mean_metrics['Sensitivity']:.4f}")
    print(f"Mean Specificity:    {mean_metrics['Specificity']:.4f}")
    print(f"Mean Precision:      {mean_metrics['Precision']:.4f}")
    print(f"Mean F1:             {mean_metrics['F1']:.4f}")
    print("="*80)
    
    return df_results, all_probs, all_targets, disease_labels


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    from config import get_config
    from models import build_model
    from dataset.build import build_loader2
    import argparse
    
    parser = argparse.ArgumentParser('Calculate metrics')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=str, default='output/metrics')
    args = parser.parse_args()
    
    # Load config
    from types import SimpleNamespace
    config_args = SimpleNamespace(
        cfg=args.cfg,
        opts=['DATA.BATCH_SIZE', str(args.batch_size)],
        batch_size=None,
        data_path=None,
        zip=None,
        cache_mode=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=None,
        disable_amp=None,
        output=None,
        tag=None,
        eval=None,
        throughput=None,
        local_rank=-1,
        amp_opt_level=None,
        pretrained=None
    )
    
    config = get_config(config_args)
    config.defrost()
    config.DATA.DATA_PATH = args.data_path
    config.DATA.BATCH_SIZE = args.batch_size
    config.MODEL.RESUME = args.checkpoint
    config.freeze()
    
    # Build dataloader
    _, _, dataset_test, _, _, data_loader_test, _ = build_loader2(config)
    
    if args.split == 'test':
        data_loader = data_loader_test
    
    # Build model
    model = build_model(config)
    model.cuda()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    # Calculate metrics
    results_df, probs, targets, labels = calculate_comprehensive_metrics(
        config, data_loader, model, output_dir=args.output
    )
    
    print(f"\nResults saved to: {args.output}")
