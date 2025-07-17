import mne
import numpy as np
import pandas as pd
import os
import glob
import scipy.stats

# --- Configuration ---
# Adjust paths based on your provided file structure
# The script itself is in 'src/eeg', so PROJECT_ROOT needs to go up 2 levels
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 

RAW_EEG_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'eeg')
PROCESSED_EEG_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'eeg')
MISC_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'misc') # For IMPORT_ME.xls etc.

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_EEG_DATA_DIR, exist_ok=True)

# Subject lists (from your MATLAB script)
PD_SX = [804, 805, 806, 807, 808, 809, 810, 811, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829]
CTL_SX = [890, 891, 892, 893, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 909, 910, 911, 912, 913, 914, 8060, 8070]
ALL_SUBJECTS = [804, 890] # making it small for the sake of testing

# Define event IDs (from your MATLAB script 'S200','S201','S202')
EVENT_ID = {
    'S200': 200,  # Target
    'S201': 201,  # Standard
    'S202': 202   # Novelty
}

TMIN, TMAX = -2.0, 2.0 # Epoching window in seconds
BASELINE_TIME = (-0.2, 0) # Baseline window for initial correction (in seconds)

# --- Channel Naming and Type Information (If your .fif files ARE labeled, this section is now simpler) ---
# Assuming your .fif files have proper channel names like 'Fp1', 'Cz', 'VEOG', 'ACCEL_X', etc.
# We no longer need to apply 'apply_channel_info' or rely on exact numerical indices for renaming.

# Identify special channels by their names (adjust if your names differ)
VEOG_CHANNEL_NAME = 'VEOG' # Or 'EOG1', 'VEOg', 'EOG_vertical', etc.
ACCEL_CHANNEL_NAMES = ['ACCEL_X', 'ACCEL_Y', 'ACCEL_Z'] # Or 'X', 'Y', 'Z', 'ACCx', etc.
STIM_CHANNEL_NAME = 'STI 014' # Common default for stimulus channels, or 'DIN', 'Trigger', etc.

# --- Helper Function for APPLE_PDDys logic ---
def apply_apple_preprocessing(raw, epochs, eeg_chans, veog_chan_name=None):
    """
    Replicates the core logic of APPLE_PDDys for a single subject.
    This will involve:
    1. Automated bad channel detection and interpolation (similar to FASTER/pop_rejchan).
    2. Automated bad epoch detection and rejection (similar to FASTER/pop_autorej).
    3. ICA for artifact correction (esp. blinks).

    Parameters:
    ----------
    raw : mne.io.Raw
        The MNE Raw object (continuous data, used for ICA fitting).
    epochs : mne.Epochs
        The MNE Epochs object to be cleaned.
    eeg_chans : list of str
        List of EEG channel names to consider for cleaning and ICA.
    veog_chan_name : str | None
        Name of the VEOG channel, if available, for ICA blink detection.

    Returns:
    -------
    epochs_cleaned : mne.Epochs
        The cleaned Epochs object.
    bad_channels : list
        List of identified bad channel names.
    bad_epochs_indices : list
        List of indices of rejected epochs.
    bad_ica_components : list
        List of identified bad ICA component indices.
    """
    print("\n--- Applying APPLE-like Pre-processing ---")

    # --- 1. Automated Bad Channel Detection & Interpolation ---
    bad_channels_detected = []
    epochs_data_eeg = epochs.get_data(picks=eeg_chans)
    
    # Criterion 1: Flat channels (std close to zero)
    channel_stds = np.std(epochs_data_eeg, axis=(0, 2))
    # MNE's own `find_bad_channels_in_epochs` can help but it's interactive.
    # Manual thresholding of std:
    flat_ch_indices = np.where(channel_stds < 1e-9)[0] # Threshold for extremely low std (flatline)
    if len(flat_ch_indices) > 0:
        flat_ch_names = [eeg_chans[i] for i in flat_ch_indices]
        print(f"Detected flat channels: {flat_ch_names}")
        bad_channels_detected.extend(flat_ch_names)

    # Criterion 2: Noisy channels (outlier std dev)
    if len(eeg_chans) > 1 and len(channel_stds) > 1: # Ensure enough channels for robust statistics
        channel_stds_z = np.abs(scipy.stats.zscore(channel_stds))
        noisy_ch_indices_z = np.where(channel_stds_z > 3)[0] # Z-score threshold of 3
        if len(noisy_ch_indices_z) > 0:
            noisy_ch_names_z = [eeg_chans[i] for i in noisy_ch_indices_z]
            print(f"Detected noisy channels (high std): {noisy_ch_names_z}")
            bad_channels_detected.extend(noisy_ch_names_z)
    
    bad_channels_final = list(np.unique(bad_channels_detected))
    
    epochs_cleaned = epochs.copy()
    if bad_channels_final:
        print(f"Identified bad channels for interpolation: {bad_channels_final}")
        # Mark bad channels in raw info and interpolate. MNE propagates this to epochs.
        raw.info['bads'].extend(bad_channels_final) # Add to existing bads if any
        raw.interpolate_bads(reset_bads=False) # Interpolate on raw data
        epochs_cleaned = epochs.copy().interpolate_bads(reset_bads=True) # Apply interpolation to epochs
    else:
        print("No bad channels identified for interpolation.")
        
    # --- 2. Automated Bad Epoch Detection & Rejection ---
    initial_reject_criteria = dict(eeg=200e-6) # 200 uV peak-to-peak for EEG
    flat_criteria = dict(eeg=1e-6) # If 1uV range or less, consider flat

    epochs_before_rejection = len(epochs_cleaned)
    epochs_cleaned.drop_bad(reject=initial_reject_criteria, flat=flat_criteria, verbose=False)
    
    # Get indices of dropped epochs by comparing original selection to current selection
    all_epoch_indices = np.arange(epochs_before_rejection)
    kept_epoch_indices = epochs_cleaned.selection
    bad_epochs_indices = np.setdiff1d(all_epoch_indices, kept_epoch_indices).tolist()
    
    print(f"Rejected {epochs_before_rejection - len(epochs_cleaned)} epochs based on amplitude thresholds and flatness.")
    
    # --- 3. ICA for Artifact Correction ---
    # Apply high-pass filter (e.g., 1 Hz) to continuous raw data for ICA stability
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

    ica = mne.preprocessing.ICA(n_components=0.99, method='picard', random_state=42, max_iter='auto')
    print("Fitting ICA (this may take a moment)...")
    ica.fit(raw_for_ica, picks=eeg_chans) # Fit ICA only on EEG channels

    # Find EOG components (blinks)
    eog_indices = []
    if veog_chan_name in raw.ch_names: # Check if VEOG channel exists in the raw data
        eog_indices, scores = ica.find_bads_eog(raw, ch_name=veog_chan_name, measure='correlation', threshold='auto', verbose=False)
        print(f"Automatically detected EOG components: {eog_indices}")
    else:
        print(f"Warning: VEOG channel '{veog_chan_name}' not found for automatic EOG detection. Manual inspection may be needed.")
    
    bad_ica_components = list(eog_indices) 
    
    if bad_ica_components:
        print(f"Excluding ICA components: {bad_ica_components} from epochs.")
        epochs_cleaned = ica.apply(epochs_cleaned.copy(), exclude=bad_ica_components)
    else:
        print("No ICA components excluded automatically.")

    print("--- APPLE-like Pre-processing Complete ---")
    return epochs_cleaned, bad_channels_final, bad_epochs_indices, bad_ica_components

# --- Main Processing Loop ---
def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw EEG data directory: {RAW_EEG_DATA_DIR}")
    print(f"Processed EEG data directory: {PROCESSED_EEG_DATA_DIR}")

    # Load demographic data (optional for subject type lookup)
    try:
        demographics_df = pd.read_excel(os.path.join(MISC_DATA_DIR, 'IMPORT_ME.xlsx'))
    except FileNotFoundError:
        print(f"Warning: IMPORT_ME.xlsx not found in {MISC_DATA_DIR}. Subject types determined by hardcoded lists.")
        demographics_df = None

    for subj_id in ALL_SUBJECTS:
        is_pd = subj_id in PD_SX
        is_ctl = subj_id in CTL_SX

        sessions_to_process = []
        if is_pd:
            sessions_to_process = [1, 2]
        elif is_ctl:
            sessions_to_process = [1]
        
        for session_num in sessions_to_process:
            # Construct filename based on the new naming convention
            # IMPORTANT: Confirm if your 'ODDBALL' data also uses '_PD_REST-epo.fif' or if it's different.
            # Assuming 'PD_REST' is the task name that changes.
            task_name = "ODDBALL" # Or "PD_REST" if all your data is rest tremor.
                                  # The MATLAB scripts referred to "ODDBALL".
            
            # This is the expected raw filename based on your new convention
            raw_fname_base = f"{subj_id}_{session_num}_PD_{task_name}" # e.g., 804_1_PD_ODDBALL
            raw_fname_suffix = "-epo.fif" # Assuming the '-epo.fif' is part of the raw continuous file name
                                          # If it's *not* continuous but already epoched, we'll adjust later.

            # Processed file name should reflect the task and pre-processing
            processed_fname = os.path.join(PROCESSED_EEG_DATA_DIR, f"{raw_fname_base}_processed{raw_fname_suffix}")
            
            # Check for existing processed file
            if os.path.exists(processed_fname):
                print(f"Skipping {subj_id}_Session{session_num}: Processed file already exists at {processed_fname}")
                continue

            print(f"\n--- Processing Subject: {subj_id}, Session: {session_num} ---")

            # --- Data Loading ---
            raw_fname_full = os.path.join(RAW_EEG_DATA_DIR, f"{raw_fname_base}{raw_fname_suffix}")
            
            # Special case for subject 810 session 1 merge
            # Assuming the 'b' file also follows the new convention: 810_1_PD_ODDBALLb-epo.fif
            if subj_id == 810 and session_num == 1:
                raw_fname_b_full = os.path.join(RAW_EEG_DATA_DIR, f"{raw_fname_base}b{raw_fname_suffix}")
                try:
                    raw_a = mne.io.read_raw_fif(raw_fname_full, preload=True, verbose=False)
                    raw_b = mne.io.read_raw_fif(raw_fname_b_full, preload=True, verbose=False)
                    raw = mne.concatenate_raws([raw_a, raw_b])
                    print(f"Merged {os.path.basename(raw_fname_full)} and {os.path.basename(raw_fname_b_full)}")
                except FileNotFoundError:
                    print(f"Error: One or both files for 810_1 merge not found. Skipping {subj_id}_{session_num}.")
                    continue
            else:
                try:
                    raw = mne.io.read_raw_fif(raw_fname_full, preload=True, verbose=False)
                    print(f"Loaded {os.path.basename(raw_fname_full)}")
                except FileNotFoundError:
                    print(f"Error: Raw file {raw_fname_full} not found. Skipping {subj_id}_{session_num}.")
                    continue
                except Exception as e:
                    print(f"Error loading {raw_fname_full}: {e}. Skipping {subj_id}_{session_num}.")
                    continue
            
            if raw is None: 
                continue # Skip if raw data could not be loaded

            # --- Channel Management (Assuming already labeled in .fif) ---
            # Now that channels are properly labeled, we directly identify by name/type.
            # No need for apply_channel_info() function
            
            # Identify EEG, EOG, and miscellaneous channels based on their type
            eeg_ch_names = raw.copy().pick_types(eeg=True, exclude='bads').ch_names # Use channels marked as EEG and not already bad
            eog_ch_names = raw.copy().pick_types(eog=True, exclude='bads').ch_names
            misc_ch_names = raw.copy().pick_types(misc=True, exclude='bads').ch_names

            # Validate that VEOG channel exists in EOG list if needed for ICA
            current_veog_channel_name = None
            if VEOG_CHANNEL_NAME in eog_ch_names:
                current_veog_channel_name = VEOG_CHANNEL_NAME
            elif len(eog_ch_names) > 0: # If VEOG_CHANNEL_NAME not exact, but some EOG exists
                current_veog_channel_name = eog_ch_names[0] # Take the first EOG channel found
                print(f"Using '{current_veog_channel_name}' as VEOG channel for ICA.")
            else:
                print("Warning: No EOG channels found. ICA EOG detection will be skipped.")

            # Drop misc channels (accelerometers) as per MATLAB script after saving if needed
            # If you need to save accelerometer data to .pkl, do it here before dropping:
            # for accel_ch in misc_ch_names:
            #     accel_data_series = raw.get_data(picks=accel_ch).flatten()
            #     # Save accel_data_series to a .pkl file here if desired
            #     # Example: pd.to_pickle(accel_data_series, os.path.join(ACCEL_DATA_DIR, f"{subj_id}_{session_num}_{accel_ch}.pkl"))
            if misc_ch_names:
                print(f"Dropping miscellaneous channels: {misc_ch_names}")
                raw.drop_channels(misc_ch_names)

            # --- Set Montage (for channel locations) ---
            montage = mne.channels.make_standard_montage('standard_1005') # or 'standard_1020'
            raw.set_montage(montage, on_missing='ignore') 
            print(f"Set montage: {montage.dig_ch_names[:5]}...")

            # --- Re-referencing to Average ---
            print(f"Applying average reference to {len(eeg_ch_names)} EEG channels.")
            # MNE automatically excludes 'eog' type channels from average reference calculation
            raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
            raw.apply_proj() # Apply the projection if not done automatically

            # --- Epoching ---
            # The MATLAB script uses 'S200', 'S201', 'S202' as event types.
            # MNE find_events automatically looks for 'STI 014' or other common stim channels.
            events = mne.find_events(raw, stim_channel=STIM_CHANNEL_NAME) 
            
            epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMAX,
                                baseline=BASELINE_TIME, preload=True, verbose=False)
            
            print(f"Epoching complete. Found {len(epochs)} epochs.")

            # --- Apply APPLE-like Pre-processing ---
            epochs_cleaned, bad_channels_found, bad_epochs_rejected_indices, bad_ica_components_found = \
                apply_apple_preprocessing(raw, epochs, eeg_ch_names, veog_chan_name=current_veog_channel_name)

            # --- Save Cleaned Epochs ---
            print(f"Saving cleaned epochs to {processed_fname}")
            epochs_cleaned.save(processed_fname, overwrite=True, verbose=False)
            
            print(f"Finished processing {subj_id}_Session{session_num}. Saved to {processed_fname}")

# --- Run the main function ---
if __name__ == '__main__':
    main()