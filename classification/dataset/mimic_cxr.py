"""
MIMIC-CXR Dataset for Multi-Label Disease Classification

This dataset loader handles the MIMIC-CXR chest X-ray dataset with multi-label
disease annotations. It loads images from CSV files with 14 disease labels and
converts grayscale X-rays to RGB format for compatibility with ImageNet-pretrained models.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR Dataset for multi-label disease classification.
    
    The dataset contains chest X-ray images with 14 disease labels based on CheXpert labeling.
    Labels can be: 1.0 (positive), 0.0 (negative), or NaN/uncertain (handled as negative).
    
    Args:
        data_root (str): Root directory containing MIMIC-CXR files
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Optional transform to be applied on images
        csv_file (str, optional): Path to CSV file with annotations. If None, uses default naming.
    """
    
    # 14 standard CheXpert disease labels used in MIMIC-CXR
    DISEASE_LABELS = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Enlarged Cardiomediastinum',
        'Fracture',
        'Lung Lesion',
        'Lung Opacity',
        'No Finding',
        'Pleural Effusion',
        'Pleural Other',
        'Pneumonia',
        'Pneumothorax',
        'Support Devices'
    ]
    
    def __init__(self, data_root, split='train', transform=None, csv_file=None):
        """
        Initialize MIMIC-CXR dataset.
        
        Args:
            data_root: Path to MIMIC-CXR root directory
            split: One of 'train', 'val', 'test'
            transform: Torchvision transforms to apply
            csv_file: Optional path to CSV file
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # Determine CSV file path
        if csv_file is None:
            # Try common naming conventions
            possible_names = [
                f'mimic-cxr-2.0.0-chexpert-{split}.csv',
                f'mimic_cxr_{split}.csv',
                f'{split}.csv',
                'mimic-cxr-2.0.0-chexpert.csv',  # Single file with split column
            ]
            
            for name in possible_names:
                potential_path = os.path.join(data_root, name)
                if os.path.exists(potential_path):
                    csv_file = potential_path
                    break
            
            if csv_file is None:
                raise FileNotFoundError(
                    f"Could not find CSV file for split '{split}' in {data_root}. "
                    f"Tried: {possible_names}"
                )
        
        print(f"Loading MIMIC-CXR {split} split from: {csv_file}")
        
        # Load CSV annotations
        self.data_df = pd.read_csv(csv_file)
        
        # Filter by split if CSV contains all splits
        if 'split' in self.data_df.columns:
            self.data_df = self.data_df[self.data_df['split'] == split].reset_index(drop=True)
        
        # Verify disease labels exist in CSV
        missing_labels = [label for label in self.DISEASE_LABELS if label not in self.data_df.columns]
        if missing_labels:
            print(f"Warning: Missing disease labels in CSV: {missing_labels}")
            print(f"Available columns: {self.data_df.columns.tolist()}")
        
        # Extract image paths - try common column names
        path_columns = ['path', 'Path', 'dicom_id', 'study_id', 'image_path']
        self.image_paths = None
        for col in path_columns:
            if col in self.data_df.columns:
                self.image_paths = self.data_df[col].values
                print(f"Using column '{col}' for image paths")
                break
        
        if self.image_paths is None:
            raise ValueError(f"Could not find image path column. Available: {self.data_df.columns.tolist()}")
        
        # Extract labels for available disease columns
        available_labels = [label for label in self.DISEASE_LABELS if label in self.data_df.columns]
        self.labels = self.data_df[available_labels].values
        
        # Handle missing labels: Convert NaN and -1 (uncertain) to 0.0 (negative)
        # This follows common practice in CheXpert/MIMIC-CXR multi-label classification
        self.labels = np.nan_to_num(self.labels, nan=0.0)
        self.labels = np.where(self.labels == -1, 0.0, self.labels)
        self.labels = self.labels.astype(np.float32)
        
        print(f"Loaded {len(self)} samples for {split} split")
        print(f"Label shape: {self.labels.shape}")
        print(f"Positive label distribution: {self.labels.sum(axis=0)}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (image, label) where image is a tensor and label is a float tensor
        """
        # Get image path
        img_path = self.image_paths[idx]
        
        # Paths from prepare_mimic_splits.py are already absolute
        # If not absolute, make it relative to data_root
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.data_root, img_path)
        
        # Load image
        try:
            # Try loading as DICOM first
            if img_path.endswith('.dcm'):
                image = self._load_dicom(img_path)
            else:
                # Load as regular image (JPG, PNG, etc.)
                image = Image.open(img_path)
                
                # Convert to grayscale if not already
                if image.mode != 'L' and image.mode != 'RGB':
                    image = image.convert('L')
            
            # Convert grayscale to RGB by repeating channels
            # This is necessary for ImageNet-pretrained models that expect 3 channels
            if image.mode == 'L':
                image = image.convert('RGB')
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (224, 224), color=0)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get labels as float tensor for BCEWithLogitsLoss
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label
    
    def _load_dicom(self, dicom_path):
        """
        Load a DICOM file and convert to PIL Image.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            PIL.Image: Grayscale image
        """
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            
            # Read DICOM file
            dcm = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            image = dcm.pixel_array
            
            # Apply VOI LUT (windowing) if available
            try:
                image = apply_voi_lut(image, dcm)
            except:
                pass
            
            # Normalize to 0-255 range
            image = image - image.min()
            image = (image / image.max() * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image, mode='L')
            
            return image
            
        except ImportError:
            print("Warning: pydicom not installed. Install with: pip install pydicom")
            raise
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            raise
    
    def get_label_names(self):
        """Return list of disease label names."""
        return self.DISEASE_LABELS
    
    def get_positive_counts(self):
        """Return count of positive samples for each disease."""
        return self.labels.sum(axis=0)
    
    def get_negative_counts(self):
        """Return count of negative samples for each disease."""
        return (1 - self.labels).sum(axis=0)
