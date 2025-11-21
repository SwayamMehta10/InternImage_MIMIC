"""
Evaluate a trained InternImage checkpoint on MIMIC-CXR test set.

Usage:
    python eval_checkpoint.py \
        --cfg configs/internimage_b_mimic_cxr_224.yaml \
        --data-path /scratch/smehta90/mimic_splits \
        --checkpoint output/mimic_cxr/internimage_b/internimage_b_mimic_cxr_224/ckpt_epoch_11.pth \
        --split test \
        --batch-size 128
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from dataset.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper


@torch.no_grad()
def validate_multilabel_single_gpu(config, data_loader, model, logger=None):
    """
    Validation function for multi-label classification without distributed training.
    
    Args:
        config: Configuration object
        data_loader: Validation data loader
        model: Model to evaluate
        logger: Logger object
        
    Returns:
        tuple: (mean_auc, loss, per_class_auc) - mean AUC-ROC, average loss, and per-class AUC-ROC
    """
    from sklearn.metrics import roc_auc_score
    
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
    
    # Concatenate all predictions and targets
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Apply sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_outputs))
    
    # Calculate per-class AUC-ROC
    per_class_auc = []
    for i in range(all_targets.shape[1]):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            per_class_auc.append(auc)
        except ValueError:
            # If only one class present in y_true, skip this class
            per_class_auc.append(0.0)
    
    # Calculate mean AUC-ROC
    mean_auc = np.mean(per_class_auc)
    
    if logger:
        logger.info(f' * Mean AUC-ROC {mean_auc:.4f} Loss {loss_meter.avg:.4f}')
        logger.info(f' * Per-class AUC-ROC: {[f"{auc:.4f}" for auc in per_class_auc]}')
    
    return mean_auc, loss_meter.avg, per_class_auc


def parse_option():
    parser = argparse.ArgumentParser('InternImage evaluation script for MIMIC-CXR', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint to evaluate')
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # dataset
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                        help='which split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for evaluation")
    
    # output
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    
    # distributed
    parser.add_argument('--local-rank', type=int, default=-1, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    # Import build_loader2 for non-distributed evaluation
    from dataset.build import build_loader2
    
    # Build dataloaders (non-distributed)
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader2(config)
    
    # Select the appropriate dataloader based on split
    if config.EVAL_SPLIT == 'train':
        data_loader = data_loader_train
        dataset = dataset_train
        split_name = 'train'
    elif config.EVAL_SPLIT == 'val':
        data_loader = data_loader_val
        dataset = dataset_val
        split_name = 'validation'
    else:  # test
        data_loader = data_loader_test
        dataset = dataset_test
        split_name = 'test'
    
    logger.info(f'Evaluating on {split_name} set with {len(dataset)} samples')

    logger.info(f'Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}')
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f'number of GFLOPs: {flops / 1e9}')

    # Load checkpoint
    logger.info(f'Loading checkpoint from {config.MODEL.RESUME}')
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
    
    # Load model weights
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(f'Checkpoint loaded: {msg}')
    logger.info(f'Checkpoint was saved at epoch {checkpoint.get("epoch", "unknown")}')
    
    if 'max_accuracy' in checkpoint:
        logger.info(f'Best validation AUC-ROC in checkpoint: {checkpoint["max_accuracy"]:.4f}')

    # Set model to evaluation mode
    model.eval()

    # Run evaluation
    logger.info(f'Starting evaluation on {split_name} set...')
    
    if config.DATA.DATASET == 'mimic_cxr':
        auc, loss, per_class_auc = validate_multilabel_single_gpu(config, data_loader, model, logger=logger)
        logger.info(f'{"=" * 60}')
        logger.info(f'{split_name.upper()} SET RESULTS:')
        logger.info(f'{"=" * 60}')
        logger.info(f'Mean AUC-ROC: {auc:.4f}')
        logger.info(f'Loss: {loss:.4f}')
        logger.info(f'Per-class AUC-ROC: {per_class_auc}')
        logger.info(f'{"=" * 60}')
        
        # Print per-class results in a nice table
        disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]
        
        logger.info(f'\nDetailed per-class results:')
        logger.info(f'{"-" * 60}')
        logger.info(f'{"Disease":<30} {"AUC-ROC":>10}')
        logger.info(f'{"-" * 60}')
        for label, auc_score in zip(disease_labels, per_class_auc):
            logger.info(f'{label:<30} {float(auc_score):>10.4f}')
        logger.info(f'{"-" * 60}')
        logger.info(f'{"Mean":<30} {auc:>10.4f}')
        logger.info(f'{"-" * 60}')
    else:
        # Standard classification evaluation
        from utils import validate
        acc1, acc5, loss = validate(config, data_loader, model)
        logger.info(f'{"=" * 60}')
        logger.info(f'{split_name.upper()} SET RESULTS:')
        logger.info(f'{"=" * 60}')
        logger.info(f'Top-1 Accuracy: {acc1:.2f}%')
        logger.info(f'Top-5 Accuracy: {acc5:.2f}%')
        logger.info(f'Loss: {loss:.4f}')
        logger.info(f'{"=" * 60}')


if __name__ == '__main__':
    _, config = parse_option()

    # Store eval split in config
    config.defrost()
    config.EVAL_SPLIT = _.split
    if _.data_path is not None:
        config.DATA.DATA_PATH = _.data_path
    if _.batch_size is not None:
        config.DATA.BATCH_SIZE = _.batch_size
    config.MODEL.RESUME = _.checkpoint
    config.freeze()

    # Set CUDA device
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
    else:
        rank = -1
        world_size = -1
    
    torch.cuda.set_device(0)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Setup logger
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, 
        name=f'{config.MODEL.NAME}_eval'
    )

    logger.info(f'Evaluation configuration:')
    logger.info(f'  Checkpoint: {config.MODEL.RESUME}')
    logger.info(f'  Split: {_.split}')
    logger.info(f'  Batch size: {config.DATA.BATCH_SIZE}')
    logger.info(f'  Data path: {config.DATA.DATA_PATH}')

    # Run evaluation
    main(config)
