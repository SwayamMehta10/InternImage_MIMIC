"""
Generate ROC curve visualizations for MIMIC-CXR multi-label classification.

Creates:
- Individual ROC curves for each disease
- Combined ROC curve plot (all diseases)
- Micro/Macro average ROC curves
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server

@torch.no_grad()
def plot_roc_curves(config, data_loader, model, output_dir='output/roc_curves'):
    """Generate ROC curve visualizations."""
    
    model.eval()
    
    # Collect all predictions and targets
    all_outputs = []
    all_targets = []
    
    print("Collecting predictions for ROC curves...")
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
    
    # Concatenate
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Apply sigmoid
    all_probs = 1 / (1 + np.exp(-all_outputs))
    
    # Disease labels
    disease_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
        'Pneumothorax', 'Support Devices'
    ]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(disease_labels))
    
    # 1. Plot individual ROC curves
    print("\nGenerating individual ROC curves...")
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    axes = axes.ravel()
    
    roc_data = {}
    
    for i, (label, color) in enumerate(zip(disease_labels, colors)):
        y_true = all_targets[:, i]
        y_prob = all_probs[:, i]
        
        if len(np.unique(y_true)) < 2:
            print(f"  Skipping {label}: only one class present")
            axes[i].text(0.5, 0.5, f'{label}\n(Only one class)', 
                        ha='center', va='center', fontsize=10)
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])
            continue
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        roc_data[label] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        # Plot
        axes[i].plot(fpr, tpr, color=color, lw=2, 
                    label=f'AUC = {roc_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate', fontsize=10)
        axes[i].set_ylabel('True Positive Rate', fontsize=10)
        axes[i].set_title(f'{label}', fontsize=12, fontweight='bold')
        axes[i].legend(loc="lower right", fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(disease_labels), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    individual_path = Path(output_dir) / 'roc_curves_individual.png'
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved individual ROC curves to: {individual_path}")
    
    # 2. Plot combined ROC curves (all on one plot)
    print("\nGenerating combined ROC curve plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, (label, color) in enumerate(zip(disease_labels, colors)):
        if label not in roc_data:
            continue
        
        fpr = roc_data[label]['fpr']
        tpr = roc_data[label]['tpr']
        roc_auc = roc_data[label]['auc']
        
        ax.plot(fpr, tpr, color=color, lw=2, alpha=0.8,
                label=f'{label} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Random (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - All Diseases (MIMIC-CXR Test Set)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_path = Path(output_dir) / 'roc_curves_combined.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined ROC curves to: {combined_path}")
    
    # 3. Plot micro-average and macro-average ROC
    print("\nGenerating micro/macro average ROC curves...")
    
    # Compute micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(all_targets.ravel(), all_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([roc_data[label]['fpr'] 
                                        for label in roc_data.keys()]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for label in roc_data.keys():
        mean_tpr += np.interp(all_fpr, roc_data[label]['fpr'], roc_data[label]['tpr'])
    
    mean_tpr /= len(roc_data)
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot individual curves (lighter)
    for label in roc_data.keys():
        ax.plot(roc_data[label]['fpr'], roc_data[label]['tpr'], 
                color='gray', lw=1, alpha=0.2)
    
    # Plot micro-average
    ax.plot(fpr_micro, tpr_micro, color='deepskyblue', lw=3,
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    
    # Plot macro-average
    ax.plot(fpr_macro, tpr_macro, color='navy', lw=3,
            label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
    
    # Plot random
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('Micro/Macro Average ROC Curves (MIMIC-CXR Test Set)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    avg_path = Path(output_dir) / 'roc_curves_average.png'
    plt.savefig(avg_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved micro/macro average ROC curves to: {avg_path}")
    
    # 4. Create summary bar chart of AUC scores
    print("\nGenerating AUC summary bar chart...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels_sorted = sorted(roc_data.keys(), key=lambda x: roc_data[x]['auc'], reverse=True)
    aucs = [roc_data[label]['auc'] for label in labels_sorted]
    colors_sorted = [colors[disease_labels.index(label)] for label in labels_sorted]
    
    bars = ax.barh(range(len(labels_sorted)), aucs, color=colors_sorted, alpha=0.8)
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=11)
    ax.set_xlabel('AUC-ROC Score', fontsize=14, fontweight='bold')
    ax.set_title('AUC-ROC Scores by Disease (MIMIC-CXR Test Set)', 
                fontsize=16, fontweight='bold')
    ax.set_xlim([0.6, 1.0])
    ax.axvline(x=np.mean(aucs), color='red', linestyle='--', lw=2, 
              label=f'Mean AUC = {np.mean(aucs):.3f}')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add value labels on bars
    for i, (bar, auc_val) in enumerate(zip(bars, aucs)):
        ax.text(auc_val + 0.005, i, f'{auc_val:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    bar_path = Path(output_dir) / 'auc_scores_bar_chart.png'
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved AUC bar chart to: {bar_path}")
    
    print(f"\n✓ All ROC visualizations saved to: {output_dir}")
    
    return roc_data


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    from config import get_config
    from models import build_model
    from dataset.build import build_loader2
    import argparse
    
    parser = argparse.ArgumentParser('Plot ROC curves')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=str, default='output/roc_curves')
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
    
    # Generate ROC curves
    roc_data = plot_roc_curves(config, data_loader, model, output_dir=args.output)
