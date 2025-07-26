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
def apply_apple_preprocessing(raw_full_channels, epochs_initial):
    """
    Replicates the core logic of APPLE_PDDys for a single subject.
    This will involve:
    1. Automated bad channel detection and interpolation (similar to FASTER/pop_rejchan).
    2. Automated bad epoch detection and rejection (similar to pop_autorej).
    3. ICA for artifact correction (esp. blinks).

    Parameters:
    ----------
    raw_full_channels : mne.io.Raw
        The MNE Raw object containing ALL original channels (EEG, EOG, Misc),
        used for ICA fitting. This object should be preloaded and pre-referenced.
    epochs_initial : mne.Epochs
        The MNE Epochs object created from raw_full_channels, containing ALL original
        channels, and will be cleaned.

    Returns:
    -------
    epochs_cleaned : mne.Epochs
        The cleaned Epochs object (still containing all channels initially).
    bad_channels : list
        List of identified bad channel names.
    bad_epochs_indices : list
        List of indices of rejected epochs.
    bad_ica_components : list
        List of identified bad ICA component indices.
    """
    print("\n--- Applying APPLE-like Pre-processing ---")
    
    # Identify EEG channels (excluding 'bads' and non-EEG types) for ICA fitting and further processing
    # This picks only 'eeg' type channels from the full raw object.
    eeg_ch_names_for_ica = raw_full_channels.copy().pick_types(eeg=True, exclude='bads').ch_names

    # --- 1. Automated Bad Channel Detection & Interpolation ---
    # Detection is done on EEG channels within initial epochs
    temp_epochs_for_bad_ch_detection = epochs_initial.copy().pick_types(eeg=True, exclude='bads')
    
    bad_channels_detected = []
    
    if temp_epochs_for_bad_ch_detection.ch_names: # Ensure there are EEG channels to check
        epochs_data_eeg = temp_epochs_for_bad_ch_detection.get_data()
        
        # Criterion 1: Flat channels (std close to zero)
        channel_stds = np.std(epochs_data_eeg, axis=(0, 2))
        flat_ch_indices = np.where(channel_stds < BAD_CH_FLAT_THRESHOLD_UV)[0] 
        if len(flat_ch_indices) > 0:
            flat_ch_names = [temp_epochs_for_bad_ch_detection.ch_names[i] for i in flat_ch_indices]
            print(f"Detected flat channels: {flat_ch_names}")
            bad_channels_detected.extend(flat_ch_names)

        # Criterion 2: Noisy channels (outlier std dev)
        if len(temp_epochs_for_bad_ch_detection.ch_names) > 1 and len(channel_stds) > 1:
            channel_stds_z = np.abs(scipy.stats.zscore(channel_stds))
            noisy_ch_indices_z = np.where(channel_stds_z > BAD_CH_NOISY_Z_THRESHOLD)[0]
            if len(noisy_ch_indices_z) > 0:
                noisy_ch_names_z = [temp_epochs_for_bad_ch_detection.ch_names[i] for i in noisy_ch_indices_z]
                print(f"Detected noisy channels (high std): {noisy_ch_names_z}")
                bad_channels_detected.extend(noisy_ch_names_z)
        
    bad_channels_final = list(np.unique(bad_channels_detected))
    
    epochs_cleaned = epochs_initial.copy() # Start with a fresh copy of input epochs
    if bad_channels_final:
        print(f"Identified bad channels for interpolation: {bad_channels_final}")
        # Add identified bads to raw_full_channels.info['bads'], ensuring no duplicates
        raw_full_channels.info['bads'].extend(bad_channels_final) 
        raw_full_channels.info['bads'] = list(np.unique(raw_full_channels.info['bads']))
        
        # Interpolate on raw (continuous data) and epochs (propagates from raw or direct on epochs)
        raw_full_channels.interpolate_bads(reset_bads=False, verbose=False) 
        epochs_cleaned = epochs_initial.copy().interpolate_bads(reset_bads=True, verbose=False) 
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
    raw_for_ica = raw_full_channels.copy().filter(l_freq=ICA_HIGH_PASS_FREQ, h_freq=None, verbose=False)

    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD, 
                                random_state=ICA_RANDOM_STATE, max_iter='auto', verbose=False)
    print("Fitting ICA (this may take a moment)...")
    # Fit ICA on EEG channels. Ensure these channels are picked from raw_for_ica
    ica.fit(raw_for_ica, picks=eeg_ch_names_for_ica) 

    # Find EOG components (blinks) - Now that VEOG is in the FIF from 00_convert_mat_to_fif.py!
    bad_ica_components = []
    if VEOG_CHANNEL_NAME in raw_full_channels.ch_names: 
        eog_indices, scores = ica.find_bads_eog(raw_full_channels, ch_name=VEOG_CHANNEL_NAME, measure='correlation', threshold='auto', verbose=False)
        if eog_indices:
            print(f"Automatically detected EOG components: {eog_indices}")
            bad_ica_components.extend(eog_indices)
        else:
            print(f"No EOG components automatically detected from '{VEOG_CHANNEL_NAME}'.")
    else:
        print(f"Warning: VEOG channel '{VEOG_CHANNEL_NAME}' not found in raw data. Skipping automatic EOG detection via dedicated channel.")
        
    # Remove duplicates if any (e.g., if multiple detection methods were used)
    bad_ica_components = list(np.unique(bad_ica_components))

    if bad_ica_components:
        print(f"Excluding ICA components: {bad_ica_components} from epochs.")
        epochs_cleaned = ica.apply(epochs_cleaned.copy(), exclude=bad_ica_components, verbose=False)
    else:
        print("No ICA components excluded automatically.")

    print("--- APPLE-like Pre-processing Complete ---")
    return epochs_cleaned, bad_channels_final, bad_epochs_indices, bad_ica_components

# --- Main Processing Loop ---
def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw EEG data directory: {RAW_EEG_DATA_DIR}")
    print(f"Processed EEG data directory: {PROCESSED_EEG_DATA_DIR}")

    os.makedirs(PROCESSED_EEG_DATA_DIR, exist_ok=True)

    try:
        demographics_df = pd.read_excel(os.path.join(MISC_DATA_DIR, 'IMPORT_ME.xlsx'))
    except FileNotFoundError:
        print(f"Warning: IMPORT_ME.xlsx not found at {os.path.join(MISC_DATA_DIR, 'IMPORT_ME.xlsx')}.")
        print("Subject type checks (PD/CTL) will rely solely on hardcoded lists.")
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
            task_name = "REST" 
            
            raw_fname_base = f"{subj_id}_{session_num}_PD_{task_name}" 
            raw_fname_suffix = RAW_FNAME_SUFFIX 

            processed_fname = os.path.join(PROCESSED_EEG_DATA_DIR, f"{raw_fname_base}_processed{raw_fname_suffix}")
            
            if os.path.exists(processed_fname) and not OVERWRITE_PROCESSED_FILES:
                print(f"Skipping {subj_id}_Session{session_num}: Processed file already exists at {processed_fname}")
                continue

            print(f"\n--- Processing Subject: {subj_id}, Session: {session_num} ---")

            # --- Data Loading ---
            raw_fname_full = os.path.join(RAW_EEG_DATA_DIR, f"{raw_fname_base}{raw_fname_suffix}")
            
            raw = None 
            if subj_id == 810 and session_num == 1:
                raw_fname_b_base = f"{subj_id}_{session_num}_PD_{task_name}b" 
                raw_fname_b_full = os.path.join(RAW_EEG_DATA_DIR, f"{raw_fname_b_base}{raw_fname_suffix}")

                try:
                    raw_a = mne.io.read_raw_fif(raw_fname_full, preload=True, verbose=False)
                    raw_b = mne.io.read_raw_fif(raw_fname_b_full, preload=True, verbose=False)
                    raw = mne.concatenate_raws([raw_a, raw_b])
                    print(f"Merged {os.path.basename(raw_fname_full)} and {os.path.basename(raw_fname_b_full)}")
                except FileNotFoundError:
                    print(f"Error: One or both files for {subj_id}_{session_num} merge not found. Skipping.")
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
            
            # --- Store Original Raw (all channels) for ICA, Montage, Re-ref ---
            # This copy will be used for operations that need all channels (like ICA fitting)
            # or for which the output will be all channels (like initial epochs).
            raw_original_all_channels = raw.copy() 

            # --- Set Montage (apply to raw_original_all_channels) ---
            try:
                montage = mne.channels.make_standard_montage('standard_1005') 
                raw_original_all_channels.set_montage(montage, on_missing='ignore') 
                print(f"Set montage.")
            except Exception as e:
                print(f"Warning: Could not set standard montage on raw_original_all_channels: {e}. Check channel names.")
                print("Proceeding without montage. Topoplots later may not work without channel locations.")

            # --- Re-referencing to Average (apply to raw_original_all_channels) ---
            if raw_original_all_channels.get_channel_types(picks='eeg'): 
                print(f"Applying average reference to EEG channels.")
                # MNE automatically excludes EOG/Misc channels from average reference calculation by default
                raw_original_all_channels.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
                raw_original_all_channels.apply_proj() 
            else:
                print("No EEG channels found for re-referencing. Skipping average reference.")

            # --- Epoching (from raw_original_all_channels) ---
            # Events are in raw_original_all_channels.annotations from 00_convert_mat_to_fif.py
            events, event_id_from_raw = mne.events_from_annotations(raw_original_all_channels, event_id=EVENT_ID)

            if len(events) > 0 and any(event[2] in EVENT_ID.values() for event in events):
                # Create epochs from the raw data that will be fed to APPLE_PDDys
                epochs_initial = mne.Epochs(raw_original_all_channels, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMAX,
                                            baseline=BASELINE_TIME, preload=True, verbose=False)
                
                print(f"Epoching complete. Found {len(epochs_initial)} epochs matching event IDs.")
                if len(epochs_initial) == 0:
                    print("Warning: No epochs created. Check event IDs and time windows. Skipping subject.")
                    continue 
            else:
                print(f"Warning: No relevant events found in raw.annotations for subject {subj_id} session {session_num}. Skipping subject.")
                continue 

            # --- Apply APPLE-like Pre-processing ---
            # Pass the raw object with all channels for ICA fitting, and the initial epochs
            epochs_cleaned, bad_channels_found, bad_epochs_rejected_indices, bad_ica_components_found = \
                apply_apple_preprocessing(raw_original_all_channels, epochs_initial) 
            
            # --- Final Channel Picking (Post-ICA Dropping as per MATLAB pipeline) ---
            # After ICA correction, pick only the EEG channels for the output
            # This mimics MATLAB's EEG.data = EEG.data(1:63,:,:) step.
            # Also ensures that only EEG channels are saved in the final processed file.
            epochs_final_eeg_only = epochs_cleaned.copy().pick_types(eeg=True)
            print(f"Final epochs object contains {len(epochs_final_eeg_only.ch_names)} EEG channels.")

            # --- Save Cleaned Epochs (EEG only) ---
            print(f"Saving cleaned epochs to {processed_fname}")
            epochs_final_eeg_only.save(processed_fname, overwrite=OVERWRITE_PROCESSED_FILES, verbose=False) 
            
            print(f"Finished processing {subj_id}_Session{session_num}. Saved to {processed_fname}")

# --- Run the main function ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR during processing: {e}")
        import traceback
        traceback.print_exc()