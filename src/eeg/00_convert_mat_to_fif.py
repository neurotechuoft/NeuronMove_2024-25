import mne
import numpy as np
import scipy.io # For loading .mat files
import os

# --- Import configurations ---
from .config import (
    PROJECT_ROOT, RAW_EEG_DATA_DIR, MISC_DATA_DIR,
    PD_SX, CTL_SX, ALL_SUBJECTS, 
    VEOG_CHANNEL_NAME, ACCEL_CHANNEL_NAMES, STIM_CHANNEL_NAME,
    RAW_FNAME_SUFFIX, OVERWRITE_PROCESSED_FILES
)

# --- Configuration for this conversion script ---
RAW_MATLAB_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'matlab_file') 
os.makedirs(RAW_MATLAB_DATA_DIR, exist_ok=True) 

# --- Main Conversion Loop ---
def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw MATLAB data directory: {RAW_MATLAB_DATA_DIR}")
    print(f"Output .fif data directory: {RAW_EEG_DATA_DIR}") 

    os.makedirs(RAW_EEG_DATA_DIR, exist_ok=True)

    for subj_id in ALL_SUBJECTS:
        is_pd = subj_id in PD_SX
        
        sessions_to_process = []
        if is_pd:
            sessions_to_process = [1, 2]
        else:
            sessions_to_process = [1]
        
        for session_num in sessions_to_process:
            mat_fname_base = f"{subj_id}_{session_num}_PD_REST" # Your confirmed .mat naming
            
            # --- IMPORTANT: Handle the _REST1.mat files ---
            # If both _REST.mat AND _REST1.mat exist, which one should be used?
            # The current loop will only try to load _REST.mat.
            # If _REST1.mat is the intended file for some or all subjects,
            # you'll need to adjust `mat_fname_base` or add logic to check for _REST1.mat.
            # For now, let's assume _REST.mat is the target for a clean run.
            # If 810 has a separate merge, that's for the _ODDBALL task.
            
            mat_fname_full = os.path.join(RAW_MATLAB_DATA_DIR, f"{mat_fname_base}.mat")
            
            fif_fname_base = f"{subj_id}_{session_num}_PD_REST" 
            fif_fname_full = os.path.join(RAW_EEG_DATA_DIR, f"{fif_fname_base}{RAW_FNAME_SUFFIX}")
            
            if os.path.exists(fif_fname_full) and not OVERWRITE_PROCESSED_FILES:
                print(f"Skipping {subj_id}_Session{session_num}: .fif file already exists at {fif_fname_full}")
                continue

            print(f"\n--- Converting Subject: {subj_id}, Session: {session_num} ---")
            print(f"Loading MATLAB file: {mat_fname_full}")

            try:
                mat_data = scipy.io.loadmat(mat_fname_full, simplify_cells=True, mat_dtype=True, squeeze_me=True)

                eeg_struct = mat_data.get('EEG', None)
                if eeg_struct is None:
                    print("Warning: 'EEG' variable not found directly. Assuming top-level variables are the EEG struct content.")
                    eeg_struct = mat_data
                
                raw_data_array = eeg_struct['data'] 
                sfreq = eeg_struct['srate']
                
                # --- FIX: Extract channel labels from list of dicts ---
                if 'chanlocs' in eeg_struct:
                    chanlocs_list = eeg_struct['chanlocs']
                    if isinstance(chanlocs_list, list) and all(isinstance(loc, dict) and 'labels' in loc for loc in chanlocs_list):
                        ch_labels_mat = [loc['labels'] for loc in chanlocs_list]
                    elif isinstance(chanlocs_list, dict) and 'labels' in chanlocs_list and isinstance(chanlocs_list['labels'], list):
                        ch_labels_mat = chanlocs_list['labels'] # Already simplified to a list
                    else:
                        raise ValueError(f"Unexpected structure for chanlocs.labels. Type: {type(chanlocs_list)}")
                else:
                    raise ValueError("Channel locations ('chanlocs') field not found in EEG struct.")
                # --- END FIX ---
                
                # Verify channel count matches data array
                if len(ch_labels_mat) != raw_data_array.shape[0]:
                    raise ValueError(f"Channel label count ({len(ch_labels_mat)}) does not match data array channel dimension ({raw_data_array.shape[0]}).")

                # Data is in uV, convert to Volts
                raw_data_array = raw_data_array * 1e-6 

                # Ensure data is channels x time points (MNE convention)
                if raw_data_array.ndim != 2: 
                    raise ValueError(f"Unexpected data dimensions: {raw_data_array.ndim}. Expected 2D (channels x time). Shape: {raw_data_array.shape}")
                
                # --- Create MNE Info object ---
                ch_types = []
                for ch_name in ch_labels_mat:
                    if ch_name == VEOG_CHANNEL_NAME: 
                        ch_types.append('eog')
                    elif ch_name in ACCEL_CHANNEL_NAMES: 
                        ch_types.append('misc')
                    elif ch_name == STIM_CHANNEL_NAME: 
                        ch_types.append('stim')
                    else: 
                        ch_types.append('eeg')

                info = mne.create_info(ch_names=ch_labels_mat, sfreq=sfreq, ch_types=ch_types)

                raw_mne = mne.io.RawArray(raw_data_array, info, verbose=False)

                # --- Add Montage (optional, but good for visualization later) ---
                try:
                    montage = mne.channels.make_standard_montage('standard_1005') 
                    raw_mne.set_montage(montage, on_missing='ignore', verbose=False) 
                    print(f"Set standard montage.")
                except Exception as e:
                    print(f"Warning: Could not set standard montage on raw_mne: {e}. Topoplots may require a custom montage.")

                # --- Convert EEGLAB Events to MNE Annotations ---
                events_eeglab = eeg_struct.get('event', [])
                if events_eeglab and isinstance(events_eeglab, (list, np.ndarray)): # Check if events_eeglab is a list/array
                    mne_annotations = []
                    for event in events_eeglab:
                        # Event properties should be directly accessible if simplify_cells=True worked as expected
                        onset_sample = event['latency'] - 1 # EEGLAB is 1-indexed, MNE is 0-indexed
                        duration_samples = 0 # Point event
                        
                        # Ensure 'type' is convertible to str and extract ID if it's like 'SXXX'
                        event_type_raw = event['type']
                        if isinstance(event_type_raw, (int, float)):
                            description = str(int(event_type_raw)) # Directly use number as string
                        elif isinstance(event_type_raw, str):
                            # Try to extract number if format is 'SXXX'
                            if event_type_raw.startswith('S') and len(event_type_raw) > 1:
                                try:
                                    description = str(int(event_type_raw[1:]))
                                except ValueError:
                                    description = event_type_raw # Fallback to raw string if conversion fails
                            else:
                                description = event_type_raw # Use raw string
                        else:
                            description = str(event_type_raw) # Fallback for other types

                        mne_annotations.append((onset_sample / sfreq, duration_samples / sfreq, description))
                    
                    if mne_annotations: # Only apply if there are annotations to add
                        raw_mne.set_annotations(mne.Annotations(
                            [a[0] for a in mne_annotations],
                            [a[1] for a in mne_annotations],
                            [a[2] for a in mne_annotations],
                            orig_time=raw_mne.info['meas_date']
                        ), verbose=False)
                        print(f"Converted {len(mne_annotations)} EEGLAB events to MNE Annotations.")
                else:
                    print("No EEGLAB events found to convert or events structure is unexpected.")

                # --- Save to .fif ---
                raw_mne.save(fif_fname_full, overwrite=OVERWRITE_PROCESSED_FILES, verbose=False)
                print(f"Successfully converted and saved to: {fif_fname_full}")

            except FileNotFoundError:
                print(f"Error: MATLAB file {mat_fname_full} not found. Skipping conversion.")
            except Exception as e:
                print(f"An error occurred during conversion for {subj_id}_{session_num}: {e}")
                import traceback
                traceback.print_exc()

# --- Run the main function ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR during processing: {e}")
        import traceback
        traceback.print_exc()