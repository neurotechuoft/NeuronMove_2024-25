import mne
import numpy as np
import pandas as pd
import os
import glob
import scipy.stats

# Import all configurations from config.py
from .config import (
    PROJECT_ROOT, RAW_EEG_DATA_DIR, PROCESSED_EEG_DATA_DIR, MISC_DATA_DIR,
    PD_SX, CTL_SX, ALL_SUBJECTS,
    EVENT_ID, TMIN, TMAX, BASELINE_TIME,
    VEOG_CHANNEL_NAME, ACCEL_CHANNEL_NAMES, STIM_CHANNEL_NAME,
    RAW_FNAME_SUFFIX, OVERWRITE_PROCESSED_FILES,
    ICA_N_COMPONENTS, ICA_METHOD, ICA_RANDOM_STATE,
    BAD_CH_FLAT_THRESHOLD_UV, BAD_CH_NOISY_Z_THRESHOLD,
    BAD_EPOCH_PEAK_TO_PEAK_UV, BAD_EPOCH_FLAT_UV,
    ICA_HIGH_PASS_FREQ
)


# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_EEG_DATA_DIR, exist_ok=True)

# --- Helper Function for APPLE_PDDys logic ---
def apply_apple_preprocessing(raw, epochs): # Removed eeg_chans, veog_chan_name as they can be inferred from raw/epochs
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
    
    # Identify EEG channels dynamically from the epochs object itself
    eeg_ch_names = epochs.copy().pick_types(eeg=True, exclude='bads').ch_names

    # --- 1. Automated Bad Channel Detection & Interpolation ---
    bad_channels_detected = []
    
    # Only proceed if there are EEG channels to check
    if eeg_ch_names:
        epochs_data_eeg = epochs.get_data(picks=eeg_ch_names) # shape: (n_epochs, n_channels, n_times)
        
        # Criterion 1: Flat channels (std close to zero)
        channel_stds = np.std(epochs_data_eeg, axis=(0, 2))
        flat_ch_indices = np.where(channel_stds < BAD_CH_FLAT_THRESHOLD_UV)[0] 
        if len(flat_ch_indices) > 0:
            flat_ch_names = [eeg_ch_names[i] for i in flat_ch_indices]
            print(f"Detected flat channels: {flat_ch_names}")
            bad_channels_detected.extend(flat_ch_names)

        # Criterion 2: Noisy channels (outlier std dev)
        if len(eeg_ch_names) > 1 and len(channel_stds) > 1:
            channel_stds_z = np.abs(scipy.stats.zscore(channel_stds))
            noisy_ch_indices_z = np.where(channel_stds_z > BAD_CH_NOISY_Z_THRESHOLD)[0]
            if len(noisy_ch_indices_z) > 0:
                noisy_ch_names_z = [eeg_ch_names[i] for i in noisy_ch_indices_z]
                print(f"Detected noisy channels (high std): {noisy_ch_names_z}")
                bad_channels_detected.extend(noisy_ch_names_z)
        
    bad_channels_final = list(np.unique(bad_channels_detected))
    
    epochs_cleaned = epochs.copy()
    if bad_channels_final:
        print(f"Identified bad channels for interpolation: {bad_channels_final}")
        # Mark bad channels in raw info and interpolate. MNE propagates this to epochs.
        raw.info['bads'].extend(bad_channels_final) # Add to existing bads if any
        # Remove duplicates from raw.info['bads'] if any
        raw.info['bads'] = list(np.unique(raw.info['bads']))
        
        raw.interpolate_bads(reset_bads=False) # Interpolate on raw data
        epochs_cleaned = epochs.copy().interpolate_bads(reset_bads=True) # Apply interpolation to epochs
    else:
        print("No bad channels identified for interpolation.")
        
    # --- 2. Automated Bad Epoch Detection & Rejection ---
    epochs_before_rejection = len(epochs_cleaned)
    epochs_cleaned.drop_bad(reject=dict(eeg=BAD_EPOCH_PEAK_TO_PEAK_UV), 
                            flat=dict(eeg=BAD_EPOCH_FLAT_UV), 
                            verbose=False)
    
    all_epoch_indices = np.arange(epochs_before_rejection)
    kept_epoch_indices = epochs_cleaned.selection
    bad_epochs_indices = np.setdiff1d(all_epoch_indices, kept_epoch_indices).tolist()
    
    print(f"Rejected {epochs_before_rejection - len(epochs_cleaned)} epochs based on amplitude thresholds and flatness.")
    
    # --- 3. ICA for Artifact Correction ---
    # Apply high-pass filter to continuous raw data for ICA stability
    raw_for_ica = raw.copy().filter(l_freq=ICA_HIGH_PASS_FREQ, h_freq=None, verbose=False)

    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD, 
                                random_state=ICA_RANDOM_STATE, max_iter='auto')
    print("Fitting ICA (this may take a moment)...")
    ica.fit(raw_for_ica, picks=eeg_ch_names) # Fit ICA only on EEG channels

    # Find EOG components (blinks)
    eog_indices = []
    # Check if VEOG_CHANNEL_NAME is actually in the raw data's current channel list
    if VEOG_CHANNEL_NAME in raw.ch_names: 
        eog_indices, scores = ica.find_bads_eog(raw, ch_name=VEOG_CHANNEL_NAME, measure='correlation', threshold='auto', verbose=False)
        print(f"Automatically detected EOG components: {eog_indices}")
    else:
        print(f"Warning: VEOG channel '{VEOG_CHANNEL_NAME}' not found in raw data for automatic EOG detection.")
        print("If you have other EOG channels, consider adding them to config.py and checking here.")
    
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

    # Create processed data directory if it doesn't exist
    os.makedirs(PROCESSED_EEG_DATA_DIR, exist_ok=True)

    # Load demographic data (optional for subject type lookup)
    try:
        demographics_df = pd.read_excel(os.path.join(MISC_DATA_DIR, 'IMPORT_ME.xlsx')) # Corrected extension
    except FileNotFoundError:
        print(f"Warning: IMPORT_ME.xlsx not found at {os.path.join(MISC_DATA_DIR, 'IMPORT_ME.xlsx')}. Subject types determined by hardcoded lists.")
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
            # Construct raw filename based on the new naming convention
            # Use 'ODDBALL' as the task_name as per MATLAB scripts context
            task_name = "ODDBALL" 
            
            raw_fname_base = f"{subj_id}_{session_num}_PD_{task_name}" 
            raw_fname_suffix = RAW_FNAME_SUFFIX # e.g., "-epo.fif"

            # Processed file name should reflect the task and pre-processing
            processed_fname = os.path.join(PROCESSED_EEG_DATA_DIR, f"{raw_fname_base}_processed{raw_fname_suffix}")
            
            # Check for existing processed file
            if os.path.exists(processed_fname) and not OVERWRITE_PROCESSED_FILES:
                print(f"Skipping {subj_id}_Session{session_num}: Processed file already exists at {processed_fname}")
                continue

            print(f"\n--- Processing Subject: {subj_id}, Session: {session_num} ---")

            # --- Data Loading ---
            raw_fname_full = os.path.join(RAW_EEG_DATA_DIR, f"{raw_fname_base}{raw_fname_suffix}")
            
            raw = None 
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
                continue 

            # --- Channel Management (Assuming already labeled in .fif) ---
            # Identify EEG, EOG, and miscellaneous channels based on their type
            eeg_ch_names_raw = raw.copy().pick_types(eeg=True, exclude='bads').ch_names 
            eog_ch_names_raw = raw.copy().pick_types(eog=True, exclude='bads').ch_names
            misc_ch_names_raw = raw.copy().pick_types(misc=True, exclude='bads').ch_names

            # Drop misc channels (accelerometers) as per MATLAB script
            if misc_ch_names_raw:
                print(f"Dropping miscellaneous channels: {misc_ch_names_raw}")
                raw.drop_channels(misc_ch_names_raw)

            # --- Set Montage (for channel locations) ---
            # Using 'standard_1005' assumes your EEG channels are 10-20 system compatible names (e.g., Fz, Cz, Pz).
            # If your EEG channel names are generic (e.g., 'EEG 001'), this will need a custom montage.
            try:
                montage = mne.channels.make_standard_montage('standard_1005') 
                raw.set_montage(montage, on_missing='ignore') 
                print(f"Set montage: {montage.dig_ch_names[:5]}...")
            except Exception as e:
                print(f"Warning: Could not set standard montage. This might be due to generic channel names or missing locations: {e}")
                print("Proceeding without montage. Topoplots later may not work without channel locations.")

            # --- Re-referencing to Average ---
            # Ensure only 'eeg' type channels are used for average reference. MNE does this by default.
            print(f"Applying average reference to {len(eeg_ch_names_raw)} EEG channels.")
            # MNE automatically excludes 'eog' type channels from average reference calculation by default
            raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
            raw.apply_proj() # Apply the projection (average reference)

            # --- Epoching ---
            # The MATLAB script used 'S200', 'S210', 'S202' as event types in pop_epoch.
            # MNE's find_events automatically looks for 'STI 014' or other common stim channels.
            events = mne.find_events(raw, stim_channel=STIM_CHANNEL_NAME) 
            
            # Epochs creation
            epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMAX,
                                baseline=BASELINE_TIME, preload=True, verbose=False)
            
            print(f"Epoching complete. Found {len(epochs)} epochs.")

            # --- Apply APPLE-like Pre-processing ---
            # Pass the name of the VEOG channel if it exists and is typed as 'eog'
            epochs_cleaned, bad_channels_found, bad_epochs_rejected_indices, bad_ica_components_found = \
                apply_apple_preprocessing(raw, epochs) # eeg_ch_names and veog_chan_name are now inferred inside

            # --- Save Cleaned Epochs ---
            print(f"Saving cleaned epochs to {processed_fname}")
            epochs_cleaned.save(processed_fname, overwrite=True, verbose=False) # Use OVERWRITE_PROCESSED_FILES constant
            
            print(f"Finished processing {subj_id}_Session{session_num}. Saved to {processed_fname}")

# --- Run the main function ---
if __name__ == '__main__':
    main()