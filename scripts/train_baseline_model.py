"""
Baseline model for Parkinson's disease classification from smartwatch data.

This script:
1. Loads preprocessed movement data and labels
2. Extracts features from time series (statistical + frequency domain)
3. Trains a simple classifier (Random Forest)
4. Evaluates performance with cross-validation

This helps assess whether the data is suitable for ML training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import scipy.stats as stats
import scipy.signal as signal
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PREP_DIR = BASE_DIR / "preprocessed"
MOV_DIR = PREP_DIR / "movement"

# Sampling rate (Hz)
FS = 100.0

# Channel structure (from preprocessing script)
TASKS = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", 
         "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]
WRISTS = ["LeftWrist", "RightWrist"]
SENSORS = ["Acceleration", "Rotation"]  # Note: preprocessing removes some sensors
AXES = ["X", "Y", "Z"]


def build_channel_list():
    """Build the channel list matching the preprocessing order."""
    channels = []
    for task in TASKS:
        for wrist in WRISTS:
            for sensor in SENSORS:
                for axis in AXES:
                    channel = f"{task}_{sensor}_{wrist}_{axis}"
                    channels.append(channel)
    return channels


def extract_time_domain_features(signal_data):
    """Extract statistical features from time series."""
    features = []
    
    # Basic statistics
    features.extend([
        np.mean(signal_data),
        np.std(signal_data),
        np.var(signal_data),
        np.median(signal_data),
        stats.skew(signal_data),
        stats.kurtosis(signal_data),
        np.min(signal_data),
        np.max(signal_data),
        np.ptp(signal_data),  # peak-to-peak
    ])
    
    # Percentiles
    features.extend([
        np.percentile(signal_data, 25),
        np.percentile(signal_data, 50),
        np.percentile(signal_data, 75),
        np.percentile(signal_data, 90),
        np.percentile(signal_data, 95),
    ])
    
    # Energy and power
    features.extend([
        np.sum(signal_data ** 2),  # energy
        np.mean(signal_data ** 2),  # mean power
    ])
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
    features.append(len(zero_crossings) / len(signal_data))
    
    return np.array(features)


def extract_frequency_domain_features(signal_data, fs=100.0):
    """Extract frequency domain features."""
    features = []
    
    # FFT
    fft_vals = np.fft.rfft(signal_data)
    fft_freq = np.fft.rfftfreq(len(signal_data), 1/fs)
    power_spectrum = np.abs(fft_vals) ** 2
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # skip DC
    features.append(fft_freq[dominant_freq_idx])
    
    # Spectral centroid (weighted mean frequency)
    if np.sum(power_spectrum) > 0:
        spectral_centroid = np.sum(fft_freq * power_spectrum) / np.sum(power_spectrum)
    else:
        spectral_centroid = 0.0
    features.append(spectral_centroid)
    
    # Power in tremor band (3-12 Hz)
    tremor_band_mask = (fft_freq >= 3) & (fft_freq <= 12)
    tremor_power = np.sum(power_spectrum[tremor_band_mask])
    total_power = np.sum(power_spectrum[1:])  # exclude DC
    if total_power > 0:
        tremor_power_ratio = tremor_power / total_power
    else:
        tremor_power_ratio = 0.0
    features.append(tremor_power_ratio)
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum_power = np.cumsum(power_spectrum)
    total_energy = cumsum_power[-1]
    if total_energy > 0:
        rolloff_idx = np.where(cumsum_power >= 0.85 * total_energy)[0]
        if len(rolloff_idx) > 0:
            features.append(fft_freq[rolloff_idx[0]])
        else:
            features.append(fft_freq[-1])
    else:
        features.append(0.0)
    
    return np.array(features)


def extract_features_from_sample(data_matrix):
    """
    Extract features from a single patient's data.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Shape (n_channels, n_timepoints) - preprocessed movement data
        
    Returns
    -------
    feature_vector : np.ndarray
        Flattened feature vector
    """
    all_features = []
    
    # Extract features from each channel
    for channel_idx in range(data_matrix.shape[0]):
        channel_data = data_matrix[channel_idx, :]
        
        # Time domain features
        time_features = extract_time_domain_features(channel_data)
        all_features.extend(time_features)
        
        # Frequency domain features
        freq_features = extract_frequency_domain_features(channel_data, FS)
        all_features.extend(freq_features)
    
    # Also compute aggregate features across all channels
    # (e.g., mean acceleration magnitude across all tasks)
    acc_channels = [i for i in range(data_matrix.shape[0]) if 'Acceleration' in build_channel_list()[i]]
    if len(acc_channels) > 0:
        acc_data = data_matrix[acc_channels, :]
        # Compute magnitude for each timepoint across all acc channels
        acc_magnitude = np.sqrt(np.sum(acc_data ** 2, axis=0))
        aggregate_features = extract_time_domain_features(acc_magnitude)
        all_features.extend(aggregate_features)
    
    return np.array(all_features)


def load_data_and_labels(use_stratified=True):
    """
    Load preprocessed data and labels.
    
    Parameters
    ----------
    use_stratified : bool
        If True, use the stratified subset. If False, use full dataset.
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    patient_ids : np.ndarray
        Patient IDs for reference
    """
    # Load file list
    if use_stratified:
        csv_path = PREP_DIR / "stratified_subset_file_list.csv"
        print(f"Loading stratified subset from: {csv_path}")
    else:
        csv_path = PREP_DIR / "file_list.csv"
        print(f"Loading full dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Map condition to label
    label_map = {
        "Healthy": 0,
        "Parkinson's": 1,
        "Other Movement Disorders": 2,
        "Essential Tremor": 2,
        "Multiple Sclerosis": 2,
        "Atypical Parkinsonism": 2,
    }
    df['label'] = df['condition'].replace(label_map)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution:")
    for label_val, label_name in [(0, "Healthy"), (1, "Parkinson's"), (2, "Other MD")]:
        count = (df['label'] == label_val).sum()
        print(f"    {label_name}: {count}")
    
    # Load data for each patient
    X_list = []
    y_list = []
    patient_ids = []
    
    print(f"\nLoading and extracting features from {len(df)} samples...")
    for idx, row in df.iterrows():
        patient_id = row['id']
        label = row['label']
        
        # Load preprocessed data
        bin_path = MOV_DIR / f"{patient_id:03d}_ml.bin"
        
        if not bin_path.exists():
            print(f"  Warning: {bin_path} not found, skipping patient {patient_id}")
            continue
        
        # Load binary data and reshape
        data = np.fromfile(bin_path, dtype=np.float32)
        # Reshape: the preprocessing script saves data as (n_channels, 976)
        # We need to infer n_channels from data size
        n_timepoints = 976
        n_channels = len(data) // n_timepoints
        
        if len(data) % n_timepoints != 0:
            print(f"  Warning: {bin_path} has unexpected size {len(data)}, skipping")
            continue
        
        data_matrix = data.reshape(n_channels, n_timepoints)
        
        # Extract features
        features = extract_features_from_sample(data_matrix)
        X_list.append(features)
        y_list.append(label)
        patient_ids.append(patient_id)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    X = np.array(X_list)
    y = np.array(y_list)
    patient_ids = np.array(patient_ids)
    
    print(f"\nFeature extraction complete!")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    return X, y, patient_ids


def main():
    """Main training and evaluation pipeline."""
    print("=" * 60)
    print("Parkinson's Disease Classification - Baseline Model")
    print("=" * 60)
    
    # Load data
    X, y, patient_ids = load_data_and_labels(use_stratified=True)
    
    # Check for NaN or Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("\nWarning: Found NaN or Inf values in features. Replacing...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train-test split (stratified)
    print("\n" + "=" * 60)
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    print("\n" + "=" * 60)
    print("Training Random Forest classifier...")
    print("  (This may take a few minutes...)")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    y_pred = rf.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Healthy", "Parkinson's", "Other MD"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Cross-validation
    print("\n" + "=" * 60)
    print("Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance (top 20)
    print("\n" + "=" * 60)
    print("Top 20 Most Important Features:")
    feature_importance = rf.feature_importances_
    top_indices = np.argsort(feature_importance)[-20:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {feature_importance[idx]:.6f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("\nAssessment:")
    if test_accuracy > 0.6:
        print("  [OK] Data appears suitable for ML training")
        print("  [OK] Model shows reasonable performance")
    elif test_accuracy > 0.5:
        print("  [WARNING] Data may be suitable but performance is modest")
        print("  [WARNING] Consider feature engineering or more complex models")
    else:
        print("  [ERROR] Model performance is poor")
        print("  [ERROR] May need more data preprocessing or different features")
    


if __name__ == "__main__":
    main()

