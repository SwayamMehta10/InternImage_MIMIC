"""
Utility functions for multi-label classification tasks.
Includes AUC-ROC calculation and validation logic for MIMIC-CXR.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from timm.utils import AverageMeter
import torch.distributed as dist


def reduce_tensor(tensor):
    """Reduce tensor across all processes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


@torch.no_grad()
def validate_multilabel(config, data_loader, model, epoch=None, logger=None):
    """
    Validation function for multi-label classification with AUC-ROC metric.
    
    Args:
        config: Configuration object
        data_loader: Validation data loader
        model: Model to evaluate
        epoch: Current epoch number (optional)
        logger: Logger object
        
    Returns:
        tuple: (mean_auc, loss) - mean AUC-ROC across all classes and average loss
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    # Collect all predictions and targets
    all_outputs = []
    all_targets = []
    
    import time
    end = time.time()
    
    for idx, (images, target) in enumerate(data_loader):
        if type(images) == list:
            images = [item.cuda(non_blocking=True) for item in images]
        else:
            images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # Forward pass
        output = model(images)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Collect predictions and targets for AUC calculation
        all_outputs.append(output.cpu())
        all_targets.append(target.cpu())
        
        # Reduce loss across GPUs
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), target.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if logger:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Mem {memory_used:.0f}MB'
                )
    
    # Check if we have any data
    if len(all_outputs) == 0:
        error_msg = (
            f"ERROR: Validation dataset is empty! "
            f"Data loader length: {len(data_loader)}, "
            f"Please check:\n"
            f"1. CSV file exists and has data\n"
            f"2. Image paths in CSV are correct\n"
            f"3. Images are accessible from the data_path"
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0).numpy()  # Shape: [N, num_classes]
    all_targets = torch.cat(all_targets, dim=0).numpy()  # Shape: [N, num_classes]
    
    # Apply sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_outputs))  # Sigmoid
    
    # Calculate AUC-ROC for each class
    num_classes = all_targets.shape[1]
    auc_scores = []
    
    for i in range(num_classes):
        targets_i = all_targets[:, i]
        probs_i = all_probs[:, i]
        
        # Check if this class has both positive and negative samples
        if len(np.unique(targets_i)) > 1:
            try:
                auc_i = roc_auc_score(targets_i, probs_i)
                auc_scores.append(auc_i)
            except ValueError as e:
                if logger:
                    logger.warning(f'Could not calculate AUC for class {i}: {e}')
                # Skip this class
                continue
        else:
            if logger:
                logger.warning(f'Class {i} has only one unique value, skipping AUC calculation')
    
    # Calculate mean AUC
    if len(auc_scores) > 0:
        mean_auc = np.mean(auc_scores)
    else:
        mean_auc = 0.0
        if logger:
            logger.warning('No valid AUC scores calculated!')
    
    # Log results
    if logger:
        if epoch is not None:
            logger.info(f'[Epoch:{epoch}] * Mean AUC-ROC {mean_auc:.4f} Loss {loss_meter.avg:.4f}')
            logger.info(f'[Epoch:{epoch}] * Per-class AUC-ROC: {[f"{auc:.4f}" for auc in auc_scores]}')
        else:
            logger.info(f' * Mean AUC-ROC {mean_auc:.4f} Loss {loss_meter.avg:.4f}')
            logger.info(f' * Per-class AUC-ROC: {[f"{auc:.4f}" for auc in auc_scores]}')
    
    return mean_auc, loss_meter.avg, auc_scores


@torch.no_grad()
def evaluate_test_set(config, data_loader, model, logger=None):
    """
    Evaluate model on test set and return detailed metrics.
    
    Args:
        config: Configuration object
        data_loader: Test data loader
        model: Model to evaluate
        logger: Logger object
        
    Returns:
        dict: Dictionary containing all metrics
    """
    mean_auc, avg_loss, per_class_auc = validate_multilabel(
        config, data_loader, model, epoch=None, logger=logger
    )
    
    results = {
        'mean_auc_roc': mean_auc,
        'avg_loss': avg_loss,
        'per_class_auc_roc': per_class_auc
    }
    
    if logger:
        logger.info('=' * 80)
        logger.info('TEST SET EVALUATION RESULTS')
        logger.info('=' * 80)
        logger.info(f'Mean AUC-ROC: {mean_auc:.4f}')
        logger.info(f'Average Loss: {avg_loss:.4f}')
        logger.info('Per-class AUC-ROC scores:')
        for i, auc in enumerate(per_class_auc):
            logger.info(f'  Class {i}: {auc:.4f}')
        logger.info('=' * 80)
    
    return results
