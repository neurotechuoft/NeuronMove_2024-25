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
# Define the directory where your raw MATLAB .mat files are located
RAW_MATLAB_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'matlab_eeg') # <--- Create this directory and place your .mat files here
os.makedirs(RAW_MATLAB_DATA_DIR, exist_ok=True) 


# --- Main Conversion Loop ---
def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw MATLAB data directory: {RAW_MATLAB_DATA_DIR}")
    print(f"Output .fif data directory: {RAW_EEG_DATA_DIR}") 

    # Create the output directory for .fif files if it doesn't exist
    os.makedirs(RAW_EEG_DATA_DIR, exist_ok=True)

    for subj_id in ALL_SUBJECTS:
        is_pd = subj_id in PD_SX
        
        sessions_to_process = []
        if is_pd:
            sessions_to_process = [1, 2]
        else: # Assumed to be CTL if not PD for session 1
            sessions_to_process = [1]
        
        for session_num in sessions_to_process:
            # Construct the name for the input MATLAB file
            # Based on your confirmed naming convention: [subjid]_[session]_PD_REST.mat
            mat_fname_base = f"{subj_id}_{session_num}_PD_REST" 
            mat_fname_full = os.path.join(RAW_MATLAB_DATA_DIR, f"{mat_fname_base}.mat")
            
            # Construct the name for the output .fif file
            # This should match what 01_preprocess_eeg.py expects
            fif_fname_base = f"{subj_id}_{session_num}_PD_REST" 
            fif_fname_full = os.path.join(RAW_EEG_DATA_DIR, f"{fif_fname_base}{RAW_FNAME_SUFFIX}") # e.g., ..._PD_REST-epo.fif
            
            if os.path.exists(fif_fname_full) and not OVERWRITE_PROCESSED_FILES:
                print(f"Skipping {subj_id}_Session{session_num}: .fif file already exists at {fif_fname_full}")
                continue

            print(f"\n--- Converting Subject: {subj_id}, Session: {session_num} ---")
            print(f"Loading MATLAB file: {mat_fname_full}")

            try:
                # Load the .mat file
                # Use h5py_default=True for newer MATLAB .mat formats (v7.3)
                mat_data = scipy.io.loadmat(mat_fname_full, simplify_cells=True, mat_dtype=True, squeeze_me=True)

                # Access the EEGLAB EEG structure
                # EEGLAB EEG struct is typically loaded as a numpy structured array or dict.
                # Common access pattern: mat_data['EEG'] or mat_data['data'] if saved directly
                # If loaded with simplify_cells=True and squeeze_me=True, it's often a dict.
                eeg_struct = mat_data.get('EEG', None) # Try to get 'EEG' variable
                if eeg_struct is None:
                    # Fallback if the data was directly saved as 'EEG' content without the 'EEG' key
                    # This depends on how the .mat was saved by MATLAB.
                    # You might need to inspect the .mat file (e.g., in MATLAB or with h5py in Python)
                    # if 'EEG' is not the top-level key.
                    print("Warning: 'EEG' variable not found directly. Assuming top-level variables are the EEG struct content.")
                    eeg_struct = mat_data # Assume the whole mat_data is the EEG struct if no 'EEG' key
                
                # --- EXTRACT DATA FROM EEGLAB EEG STRUCTURE ---
                # Data array: EEG.data, should be (channels x time) for continuous data
                raw_data_array = eeg_struct['data'] 
                
                # Sampling rate: EEG.srate
                sfreq = eeg_struct['srate']
                
                # Channel names: EEG.chanlocs.labels
                # chanlocs is an array of structs, each with a 'labels' field
                # Need to handle structure of chanlocs and labels carefully
                if 'chanlocs' in eeg_struct and 'labels' in eeg_struct['chanlocs']:
                    # If chanlocs is an array of dicts/structs:
                    if isinstance(eeg_struct['chanlocs'], np.ndarray): # Array of chanlocs structs
                        ch_labels_mat = [loc['labels'] for loc in eeg_struct['chanlocs']]
                    else: # Single chanlocs struct
                         ch_labels_mat = eeg_struct['chanlocs']['labels']
                else:
                    raise ValueError("Channel labels (chanlocs.labels) not found in EEG struct.")
                
                # Event information: EEG.event
                # EEG.event is an array of structs, each with 'type' and 'latency' (in samples)
                events_eeglab = eeg_struct.get('event', []) # Get 'event' field, default to empty list if not found
                
                # --- Convert Data to Volts if it's in microvolts (uV) ---
                # EEGLAB data is often in microvolts. MNE expects Volts.
                # Assuming data is in uV:
                raw_data_array = raw_data_array * 1e-6 

                # Ensure data is channels x time points (MNE convention)
                if raw_data_array.ndim == 2: # Continuous data (channels x time)
                    # MNE expects (n_channels, n_times). EEGLAB data is often already in this format.
                    pass # No transpose needed if already (channels x time)
                elif raw_data_array.ndim == 3: # If it's channels x samples x trials (epoched)
                    raise ValueError(f"Data is 3D (epoched). This script expects continuous raw data. Shape: {raw_data_array.shape}")
                else:
                    raise ValueError(f"Unexpected data dimensions: {raw_data_array.ndim}. Expected 2D (channels x time).")
                
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

                # --- Create MNE Raw object ---
                raw_mne = mne.io.RawArray(raw_data_array, info, verbose=False)

                # --- Add Montage (optional, but good for visualization later) ---
                # This assumes EEG channel names are standard (e.g., 'Fz', 'Pz').
                # If they are not, this block will throw a warning, but raw_mne will still be valid.
                try:
                    # Filter montage channels to only those present in the data
                    montage = mne.channels.make_standard_montage('standard_1005') 
                    # Set montage only for the channels in the data
                    raw_mne.set_montage(montage, on_missing='ignore', verbose=False) 
                    print(f"Set standard montage.")
                except Exception as e:
                    print(f"Warning: Could not set standard montage on raw_mne: {e}")
                    print("This might be due to custom channel names. Topoplots may require a custom montage.")

                # --- Convert EEGLAB Events to MNE Events ---
                if events_eeglab:
                    # EEGLAB events usually have 'type' (string) and 'latency' (sample index)
                    # MNE events are a NumPy array: (n_events, 3) with (onset_sample, duration_samples, event_id)
                    # We need to map EEGLAB event types (strings/numbers) to MNE event IDs (integers).
                    # For simple cases like 'S1', 'S2', etc., map to integer representations.
                    # The original MATLAB script for rest data uses triggers 1,2,3,4.
                    # Let's map them directly to themselves as event_ids.
                    
                    # Create a mapping for event IDs (if needed, otherwise directly use int values)
                    # Assuming EEG.event.type is already numerical or can be converted to int.
                    # Or a more robust mapping: {'S3': 3, 'S4': 4, 'S1': 1, 'S2': 2}
                    
                    mne_events = []
                    # EEGLAB event latency is 1-indexed, MNE is 0-indexed. Adjust.
                    for event in events_eeglab:
                        # Ensure 'type' is convertible to int if it's a string like 'S1'
                        event_type_str = str(event['type']) # Convert to string to check first char
                        event_id_int = int(event_type_str[1:]) if event_type_str.startswith('S') else int(event_type_str)

                        # Duration is often 0 for point events in MNE
                        mne_events.append([event['latency'] - 1, 0, event_id_int]) 
                    
                    raw_mne.set_annotations(mne.Annotations(
                        [e[0]/sfreq for e in mne_events], # onset in seconds
                        [e[1]/sfreq for e in mne_events], # duration in seconds
                        [str(e[2]) for e in mne_events], # description (string)
                        orig_time=raw_mne.info['meas_date'] # use measurement date from raw info
                    ), verbose=False)
                    print(f"Converted {len(mne_events)} EEGLAB events to MNE Annotations.")
                else:
                    print("No EEGLAB events found to convert.")


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