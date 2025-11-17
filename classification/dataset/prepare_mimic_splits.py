"""
Prepare MIMIC-CXR train/val/test split CSV files from official files.

This script combines:
- mimic-cxr-2.0.0-chexpert.csv.gz (labels) - from shared folder
- mimic-cxr-2.0.0-split.csv (train/val/test assignments) - download to your directory

And generates separate CSV files for each split with corrected image paths and labels.
"""

import pandas as pd
import os
import sys


def prepare_mimic_splits(shared_data_root, split_file_path, output_dir):
    """
    Prepare train/val/test CSV files from official MIMIC-CXR files.
    
    Args:
        shared_data_root: Shared folder with images (/scratch/aalla4/shared_folder/MIMIC)
        split_file_path: Path to mimic-cxr-2.0.0-split.csv (in your directory)
        output_dir: Directory to save output CSV files (your directory)
    """
    print("=" * 60)
    print("Preparing MIMIC-CXR train/val/test splits")
    print("=" * 60)
    
    # Check if files exist
    chexpert_file = os.path.join(shared_data_root, 'mimic-cxr-2.0.0-chexpert.csv.gz')
    
    if not os.path.exists(chexpert_file):
        print(f"ERROR: {chexpert_file} not found!")
        return False
    
    if not os.path.exists(split_file_path):
        print(f"ERROR: {split_file_path} not found!")
        print("Download from: https://physionet.org/content/mimic-cxr-jpg/2.1.0/")
        print("Run: wget https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz")
        print("     gunzip mimic-cxr-2.0.0-split.csv.gz")
        return False
    
    print(f"Loading {chexpert_file}...")
    chexpert_df = pd.read_csv(chexpert_file, compression='gzip')
    print(f"  Shape: {chexpert_df.shape}")
    print(f"  Columns: {chexpert_df.columns.tolist()}")
    
    print(f"\nLoading {split_file_path}...")
    split_df = pd.read_csv(split_file_path)
    print(f"  Shape: {split_df.shape}")
    print(f"  Columns: {split_df.columns.tolist()}")
    
    # Merge on subject_id, study_id
    print("\nMerging labels with split assignments...")
    merged_df = pd.merge(
        chexpert_df,
        split_df,
        on=['subject_id', 'study_id'],
        how='inner'
    )
    print(f"  Merged shape: {merged_df.shape}")
    
    # Build image paths in MIMIC-CXR format: pXX/pXXXXXXXX/sXXXXXXXX/XXXXXXXX.jpg
    # The path format is: files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    print("\nBuilding image paths...")
    
    def build_path(row):
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        
        # Handle dicom_id - it might be in different columns
        if 'dicom_id' in row:
            dicom_id = str(row['dicom_id'])
        else:
            # If no dicom_id, we'll need to handle this differently
            dicom_id = 'unknown'
        
        # Build absolute path to shared folder
        path = os.path.join(shared_data_root, f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg")
        return path
    
    merged_df['path'] = merged_df.apply(build_path, axis=1)
    
    # Split into train/val/test
    splits = ['train', 'validate', 'test']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        split_subset = merged_df[merged_df['split'] == split].copy()
        
        # For validation split, rename to 'val' for consistency
        output_split_name = 'val' if split == 'validate' else split
        
        output_file = os.path.join(output_dir, f'mimic-cxr-2.0.0-chexpert-{output_split_name}.csv')
        
        print(f"\nSaving {output_split_name} split...")
        print(f"  Samples: {len(split_subset)}")
        print(f"  Output: {output_file}")
        
        split_subset.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("SUCCESS! Split files created in:", output_dir)
    print("  - mimic-cxr-2.0.0-chexpert-train.csv")
    print("  - mimic-cxr-2.0.0-chexpert-val.csv")
    print("  - mimic-cxr-2.0.0-chexpert-test.csv")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        shared_data_root = sys.argv[1]
        split_file_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else os.getcwd()
    else:
        print("Usage: python prepare_mimic_splits.py <shared_data_root> <split_file_path> [output_dir]")
        print()
        print("Example:")
        print("  python prepare_mimic_splits.py \\")
        print("    /scratch/aalla4/shared_folder/MIMIC \\")
        print("    /scratch/smehta90/mimic-cxr-2.0.0-split.csv \\")
        print("    /scratch/smehta90/mimic_splits")
        sys.exit(1)
    
    print(f"Shared data root: {shared_data_root}")
    print(f"Split file: {split_file_path}")
    print(f"Output directory: {output_dir}")
    
    success = prepare_mimic_splits(shared_data_root, split_file_path, output_dir)
    
    if not success:
        sys.exit(1)
