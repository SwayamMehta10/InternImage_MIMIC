"""
Verify MIMIC-CXR dataset completeness by checking file counts.

This script checks if the shared folder has the expected number of images
based on the official MIMIC-CXR-JPG dataset statistics.
"""

import os
import sys
from pathlib import Path


def count_images(data_root):
    """
    Count JPG images in the MIMIC-CXR files directory.
    
    Args:
        data_root: Path to MIMIC-CXR root (containing 'files' subdirectory)
    """
    files_dir = os.path.join(data_root, 'files')
    
    if not os.path.exists(files_dir):
        print(f"ERROR: {files_dir} does not exist!")
        return None
    
    print(f"Scanning {files_dir}...")
    print("This may take a few minutes...\n")
    
    # Count JPG files
    jpg_count = 0
    patient_dirs = []
    
    # Iterate through patient directories (p10, p11, ..., p19)
    for patient_prefix in os.listdir(files_dir):
        patient_prefix_path = os.path.join(files_dir, patient_prefix)
        
        if not os.path.isdir(patient_prefix_path):
            continue
            
        if not patient_prefix.startswith('p'):
            continue
        
        patient_dirs.append(patient_prefix)
        
        # Count files in this prefix directory
        for root, dirs, files in os.walk(patient_prefix_path):
            jpg_files = [f for f in files if f.endswith('.jpg')]
            jpg_count += len(jpg_files)
    
    return jpg_count, sorted(patient_dirs)


def main():
    """Main function to verify dataset completeness."""
    
    # Expected statistics for MIMIC-CXR-JPG 2.0.0
    EXPECTED_TOTAL_IMAGES = 377110  # Official count
    EXPECTED_PATIENT_PREFIXES = ['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19']
    
    print("=" * 70)
    print("MIMIC-CXR Dataset Verification")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        # Check both common locations
        locations = [
            '/scratch/aalla4/shared_folder/MIMIC',
            '/data/jliang12/jpang12/dataset/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0'
        ]
        
        print("Checking both available shared folders:\n")
        
        for loc in locations:
            print(f"\n{'='*70}")
            print(f"Location: {loc}")
            print('='*70)
            
            if not os.path.exists(loc):
                print(f"⚠ Path does not exist or not accessible")
                continue
            
            result = count_images(loc)
            
            if result is None:
                continue
                
            jpg_count, patient_dirs = result
            
            print(f"\n📊 Results for {loc}:")
            print(f"   Patient directories found: {patient_dirs}")
            print(f"   Total JPG images: {jpg_count:,}")
            print(f"   Expected images: {EXPECTED_TOTAL_IMAGES:,}")
            
            if jpg_count == EXPECTED_TOTAL_IMAGES:
                print(f"   ✅ Dataset is COMPLETE!")
            else:
                diff = EXPECTED_TOTAL_IMAGES - jpg_count
                percentage = (jpg_count / EXPECTED_TOTAL_IMAGES) * 100
                print(f"   ⚠ Missing {diff:,} images ({percentage:.1f}% complete)")
            
            # Check patient directories
            missing_prefixes = set(EXPECTED_PATIENT_PREFIXES) - set(patient_dirs)
            if missing_prefixes:
                print(f"   ⚠ Missing patient prefixes: {sorted(missing_prefixes)}")
            else:
                print(f"   ✅ All patient directories present")
        
        return
    
    # Single location check
    data_root = sys.argv[1]
    print(f"Checking: {data_root}\n")
    
    result = count_images(data_root)
    
    if result is None:
        sys.exit(1)
    
    jpg_count, patient_dirs = result
    
    print(f"\n{'='*70}")
    print("📊 Results:")
    print('='*70)
    print(f"Patient directories found: {patient_dirs}")
    print(f"Total JPG images: {jpg_count:,}")
    print(f"Expected images: {EXPECTED_TOTAL_IMAGES:,}")
    
    if jpg_count == EXPECTED_TOTAL_IMAGES:
        print("\n✅ Dataset is COMPLETE!")
        print("You can proceed with training.")
    else:
        diff = EXPECTED_TOTAL_IMAGES - jpg_count
        percentage = (jpg_count / EXPECTED_TOTAL_IMAGES) * 100
        print(f"\n⚠ Missing {diff:,} images ({percentage:.1f}% complete)")
        
        if percentage >= 95:
            print("Dataset is mostly complete and should work for training.")
        else:
            print("⚠ WARNING: Significant portion of dataset is missing.")
    
    # Check patient directories
    missing_prefixes = set(EXPECTED_PATIENT_PREFIXES) - set(patient_dirs)
    if missing_prefixes:
        print(f"\n⚠ Missing patient prefixes: {sorted(missing_prefixes)}")
    else:
        print("\n✅ All patient directories (p10-p19) are present")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
