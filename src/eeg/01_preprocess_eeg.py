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
def apply_apple_preprocessing(raw, epochs):
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
    
    # Identify EEG channels (excluding 'bads' and non-EEG types) for ICA fitting and further processing
    # This picks only 'eeg' type channels.
    eeg_ch_names_for_ica = raw.copy().pick_types(eeg=True, exclude='bads').ch_names

    # --- 1. Automated Bad Channel Detection & Interpolation ---
    # Detection is done on EEG channels within epochs
    temp_epochs_for_bad_ch_detection = epochs.copy().pick_types(eeg=True, exclude='bads')
    
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
    
    epochs_cleaned = epochs.copy() 
    if bad_channels_final:
        print(f"Identified bad channels for interpolation: {bad_channels_final}")
        # Add identified bads to raw.info['bads'], ensuring no duplicates
        raw.info['bads'].extend(bad_channels_final) 
        raw.info['bads'] = list(np.unique(raw.info['bads']))
        
        # Interpolate on raw (continuous data) and epochs
        raw.interpolate_bads(reset_bads=False, verbose=False) 
        epochs_cleaned = epochs.copy().interpolate_bads(reset_bads=True, verbose=False) 
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
                                random_state=ICA_RANDOM_STATE, max_iter='auto', verbose=False)
    print("Fitting ICA (this may take a moment)...")
    # Fit ICA on EEG channels. Ensure these channels are picked from raw_for_ica
    ica.fit(raw_for_ica, picks=eeg_ch_names_for_ica) 

    # Find EOG components (blinks) - Now that VEOG is in the FIF!
    bad_ica_components = []
    if VEOG_CHANNEL_NAME in raw.ch_names: 
        eog_indices, scores = ica.find_bads_eog(raw, ch_name=VEOG_CHANNEL_NAME, measure='correlation', threshold='auto', verbose=False)
        if eog_indices:
            print(f"Automatically detected EOG components: {eog_indices}")
            bad_ica_components.extend(eog_indices)
        else:
            print(f"No EOG components automatically detected from '{VEOG_CHANNEL_NAME}'.")
    else:
        print(f"Warning: VEOG channel '{VEOG_CHANNEL_NAME}' not found in raw data. Skipping automatic EOG detection via dedicated channel.")
        
    # Remove duplicates if any (e.g., if manually added components later)
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

            # --- Channel Management: Identify and Drop Aux Channels as per MATLAB pipeline ---
            # MATLAB: EEG.VEOG=squeeze(EEG.data(64,:,:)); EEG.X/Y/Z=squeeze(EEG.data(65-67,:,:));
            #         EEG.data=EEG.data(1:63,:,:); EEG.nbchan=63;
            # This means EOG and ACCEL channels were removed from the main EEG.data.
            
            # Identify channels by their MNE type
            eog_ch_names_raw = raw.copy().pick_types(eog=True, exclude='bads').ch_names
            misc_ch_names_raw = raw.copy().pick_types(misc=True, exclude='bads').ch_names
            stim_ch_names_raw = raw.copy().pick_types(stim=True, exclude='bads').ch_names # Keep stim for events

            # Channels to be dropped from the main raw object AFTER useful information (like EOG for ICA) is used
            # For now, we'll keep stim channel as it's needed for mne.find_events.
            # Accel are dropped as they weren't used in further MATLAB steps for EEG.
            channels_to_drop_post_processing = []
            channels_to_drop_post_processing.extend(misc_ch_names_raw) # Drop accelerometers
            # If VEOG is not needed for future stages of analysis (only for ICA artifact detection)
            # and if MATLAB also dropped it from the main EEG, then add it here.
            # MATLAB did store EEG.VEOG but then dropped it from EEG.data for ICA.
            channels_to_drop_post_processing.extend(eog_ch_names_raw) 
            
            # --- Set Montage (for channel locations) ---
            # Apply montage to all channels, MNE will handle types internally
            try:
                montage = mne.channels.make_standard_montage('standard_1005') 
                raw.set_montage(montage, on_missing='ignore') 
                print(f"Set montage: {montage.dig_ch_names[:5]}...")
            except Exception as e:
                print(f"Warning: Could not set standard montage: {e}. Check channel names.")
                print("Proceeding without montage. Topoplots later may not work without channel locations.")

            # --- Re-referencing to Average ---
            # Apply average reference to EEG channels. MNE automatically excludes EOG/Misc from this.
            if raw.get_channel_types(picks='eeg'): 
                print(f"Applying average reference to EEG channels.")
                raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
                raw.apply_proj() 
            else:
                print("No EEG channels found for re-referencing. Skipping average reference.")


            # --- Epoching ---
            if STIM_CHANNEL_NAME in raw.ch_names:
                events = mne.find_events(raw, stim_channel=STIM_CHANNEL_NAME) 
                
                if len(events) > 0 and any(event[2] in EVENT_ID.values() for event in events):
                    epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMAX,
                                        baseline=BASELINE_TIME, preload=True, verbose=False)
                    
                    print(f"Epoching complete. Found {len(epochs)} epochs matching event IDs.")
                    if len(epochs) == 0:
                        print("Warning: No epochs created. Check event IDs and time windows. Skipping subject.")
                        continue 
                else:
                    print(f"Warning: No relevant events found in '{STIM_CHANNEL_NAME}' for subject {subj_id} session {session_num}. Skipping subject.")
                    continue 
            else:
                print(f"Warning: Stimulus channel '{STIM_CHANNEL_NAME}' not found in raw data. Skipping subject.")
                continue 

            # --- Apply APPLE-like Pre-processing ---
            # Pass the raw object (with all channels including EOG) and the epoched data
            epochs_cleaned, bad_channels_found, bad_epochs_rejected_indices, bad_ica_components_found = \
                apply_apple_preprocessing(raw, epochs) 
            
            # --- Post-ICA Channel Dropping (as per MATLAB pipeline) ---
            # MATLAB dropped VEOG, X, Y, Z from the main EEG.data AFTER ICA.
            # Here, we create a new Epochs object containing only EEG channels
            # for consistency with MATLAB's final EEG.data structure.
            epochs_final = epochs_cleaned.copy().pick_types(eeg=True)
            print(f"Final epochs object contains {len(epochs_final.ch_names)} EEG channels.")


            # --- Save Cleaned Epochs ---
            print(f"Saving cleaned epochs to {processed_fname}")
            # Save the epochs_final (EEG only) for next stages
            epochs_final.save(processed_fname, overwrite=OVERWRITE_PROCESSED_FILES, verbose=False) 
            
            print(f"Finished processing {subj_id}_Session{session_num}. Saved to {processed_fname}")

# --- Run the main function ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR during processing: {e}")
        import traceback
        traceback.print_exc()