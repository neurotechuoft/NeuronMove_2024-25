# pipeline configuration for eeg deep learning model
# Here, all the global variables, project root directories, and constants are defined

import os

import os

# --- Project Root and Core Directories ---

# Get the directory of the current file (config.py is in src/eeg/)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the project root (NTUT25_SOFTWARE)
# _current_dir is '.../NTUT25_SOFTWARE/src/eeg'
# os.path.join(_current_dir, '..', '..') takes us to '.../NTUT25_SOFTWARE'
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..'))

# Define key data directories relative to PROJECT_ROOT
RAW_EEG_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'eeg', 'raw')
PROCESSED_EEG_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'eeg', 'processed')
CLASSIFIER_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'eeg', 'classifier_outputs') # As per your directory structure
MISC_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'misc')
ACCELEROMETER_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'accelerometer') # As per your directory structure

# --- Subject Lists ---
PD_SX = [804, 805, 806, 807, 808, 809, 810, 811, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829]
CTL_SX = [890, 891, 892, 893, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 909, 910, 911, 912, 913, 914, 8060, 8070]
ALL_SUBJECTS = [804, 890]

# --- EEG Processing Parameters ---
EVENT_ID = {
    'S200': 200,  # Target
    'S201': 201,  # Standard
    'S202': 202   # Novelty
}

TMIN, TMAX = -2.0, 2.0 # Epoching window in seconds
BASELINE_TIME = (-0.2, 0) # Baseline window for initial correction (in seconds)

# Channel Names (assuming these are the actual names in your .fif files)
VEOG_CHANNEL_NAME = 'VEOG' # Example: check your .fif raw.info['ch_names']
ACCEL_CHANNEL_NAMES = ['ACCEL_X', 'ACCEL_Y', 'ACCEL_Z'] # Example: check your .fif raw.info['ch_names']
STIM_CHANNEL_NAME = 'STI 014' # Example: check your .fif raw.info['ch_names'] or mne.find_events output

# --- File Naming Conventions ---
# Base name for raw EEG files (e.g., "801_1_PD_ODDBALL" or "801_1_PD_REST")
# The variable 'task_name' will be set in the main script loop (e.g., "ODDBALL" or "REST")
RAW_FNAME_SUFFIX = "-epo.fif" # Assuming this suffix means it's a raw continuous file for now

# --- Processing Flags / Options ---
# Set to True to overwrite existing processed files without warning
OVERWRITE_PROCESSED_FILES = False 

# --- ICA Parameters ---
ICA_N_COMPONENTS = 0.99 # Number of components to retain (e.g., 99% variance explained)
ICA_METHOD = 'picard'
ICA_RANDOM_STATE = 42 # For reproducibility of ICA results

# --- Automated Bad Channel/Epoch Detection Thresholds ---
BAD_CH_FLAT_THRESHOLD_UV = 1e-9 # Standard deviation below this is flatline (in Volts)
BAD_CH_NOISY_Z_THRESHOLD = 3    # Z-score of channel standard deviation above this is noisy
BAD_EPOCH_PEAK_TO_PEAK_UV = 200e-6 # Peak-to-peak amplitude above this is bad (in Volts)
BAD_EPOCH_FLAT_UV = 1e-6 # Flatness threshold for epochs (in Volts, range 0-peak)

# --- Filtering Parameters ---
ICA_HIGH_PASS_FREQ = 1.0 # High-pass filter for ICA fitting (in Hz)