"""
Quick validation script to check if data is suitable for training.

This script performs basic checks without requiring sklearn,
to quickly assess data quality and structure.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PREP_DIR = BASE_DIR / "preprocessed"
MOV_DIR = PREP_DIR / "movement"

def validate_data():
    """Validate that data can be loaded and has suitable structure for training."""
    print("=" * 60)
    print("Data Validation for ML Training")
    print("=" * 60)
    
    # Check if files exist
    print("\n1. Checking file structure...")
    strat_csv = PREP_DIR / "stratified_subset_file_list.csv"
    full_csv = PREP_DIR / "file_list.csv"
    
    if strat_csv.exists():
        df = pd.read_csv(strat_csv)
        print(f"   [OK] Stratified subset found: {len(df)} samples")
    elif full_csv.exists():
        df = pd.read_csv(full_csv)
        print(f"   [OK] Full dataset found: {len(df)} samples")
    else:
        print("   [ERROR] No dataset CSV found!")
        return False
    
    # Check label distribution
    print("\n2. Checking label distribution...")
    label_map = {
        "Healthy": 0,
        "Parkinson's": 1,
        "Other Movement Disorders": 2,
        "Essential Tremor": 2,
        "Multiple Sclerosis": 2,
        "Atypical Parkinsonism": 2,
    }
    df['label'] = df['condition'].replace(label_map)
    
    label_counts = df['label'].value_counts().sort_index()
    print(f"   Label distribution:")
    for label_val, label_name in [(0, "Healthy"), (1, "Parkinson's"), (2, "Other MD")]:
        count = label_counts.get(label_val, 0)
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"     {label_name}: {count} ({pct:.1f}%)")
    
    # Check if data files exist
    print("\n3. Checking data files...")
    missing_files = []
    sample_sizes = []
    
    sample_count = min(10, len(df))  # Check first 10 samples
    print(f"   Checking first {sample_count} samples...")
    
    for idx, row in df.head(sample_count).iterrows():
        patient_id = row['id']
        bin_path = MOV_DIR / f"{patient_id:03d}_ml.bin"
        
        if not bin_path.exists():
            missing_files.append(patient_id)
        else:
            # Check file size and structure
            data = np.fromfile(bin_path, dtype=np.float32)
            n_timepoints = 976
            n_channels = len(data) // n_timepoints
            
            if len(data) % n_timepoints == 0:
                sample_sizes.append((n_channels, n_timepoints))
            else:
                print(f"     ⚠ Warning: {patient_id} has unexpected size {len(data)}")
    
    if missing_files:
        print(f"   [ERROR] Missing files for {len(missing_files)} patients")
        return False
    else:
        print(f"   [OK] All checked files exist")
    
    # Check data consistency
    print("\n4. Checking data consistency...")
    if sample_sizes:
        n_channels_set = set(s[0] for s in sample_sizes)
        n_timepoints_set = set(s[1] for s in sample_sizes)
        
        if len(n_channels_set) == 1 and len(n_timepoints_set) == 1:
            n_channels, n_timepoints = sample_sizes[0]
            print(f"   [OK] Consistent structure: {n_channels} channels x {n_timepoints} timepoints")
            print(f"   [OK] Total features per sample: {n_channels * n_timepoints} values")
        else:
            print(f"   [WARNING] Inconsistent data shapes detected")
            print(f"     Channels: {n_channels_set}")
            print(f"     Timepoints: {n_timepoints_set}")
    
    # Check for NaN/Inf in sample
    print("\n5. Checking data quality...")
    sample_id = df.iloc[0]['id']
    sample_path = MOV_DIR / f"{sample_id:03d}_ml.bin"
    if sample_path.exists():
        sample_data = np.fromfile(sample_path, dtype=np.float32)
        n_timepoints = 976
        n_channels = len(sample_data) // n_timepoints
        sample_matrix = sample_data.reshape(n_channels, n_timepoints)
        
        has_nan = np.any(np.isnan(sample_matrix))
        has_inf = np.any(np.isinf(sample_matrix))
        
        if not has_nan and not has_inf:
            print(f"   [OK] No NaN or Inf values detected")
        else:
            print(f"   [WARNING] NaN or Inf values found")
        
        # Check value ranges
        data_min, data_max = np.min(sample_matrix), np.max(sample_matrix)
        print(f"   [OK] Data range: [{data_min:.4f}, {data_max:.4f}]")
    
    # Summary assessment
    print("\n" + "=" * 60)
    print("ASSESSMENT SUMMARY")
    print("=" * 60)
    
    issues = []
    if len(df) < 50:
        issues.append("Small dataset (< 50 samples)")
    
    min_class_size = label_counts.min() if len(label_counts) > 0 else 0
    if min_class_size < 10:
        issues.append(f"Very small class size (min: {min_class_size})")
    
    if len(label_counts) < 3:
        issues.append("Missing classes")
    
    if issues:
        print("\n[WARNING] Potential Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n[OK] No major issues detected")
    
    print("\n[OK] Data appears structurally suitable for training")
    print("\nRecommendations:")
    print("  1. Install required packages: pip install -r requirements.txt")
    print("  2. Run train_baseline_model.py to train a baseline model")
    print("  3. Consider using stratified subset for balanced evaluation")
    print("  4. Feature engineering may improve performance")
    
    return True


if __name__ == "__main__":
    validate_data()

