import mne
import numpy as np
import os
import pandas as pd
import pickle
from collections import defaultdict

# Import all configurations
from . import config 

# --- New Helper Function: Get all subject/session/type combinations ---
def get_all_subject_combinations(pd_sx, ctl_sx):
    """
    Generates a list of all subject/session/type combinations to process.
    This replaces the logic from 01_preprocess_eeg.py's main loop for consistency.
    """
    combinations = []
    for subj_id in pd_sx:
        combinations.append((subj_id, 1, 'PD_ON'))
        combinations.append((subj_id, 2, 'PD_OFF'))
    for subj_id in ctl_sx:
        combinations.append((subj_id, 1, 'CTL'))
    return combinations

def main():
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Processed EEG data directory: {config.PROCESSED_EEG_DATA_DIR}")

    aggregated_data_dir = os.path.join(config.PROCESSED_EEG_DATA_DIR, 'aggregated')
    os.makedirs(aggregated_data_dir, exist_ok=True)
    
    output_fname = os.path.join(aggregated_data_dir, config.AGGREGATED_FNAME)

    if os.path.exists(output_fname) and not config.OVERWRITE_PROCESSED_FILES:
        print(f"Aggregated file already exists at {output_fname}. Skipping.")
        return

    print("\n--- Starting Data Aggregation and Trial Matching ---")
    
    # --- Step 1: Load all available processed epochs and store them in a single dict ---
    all_loaded_epochs = {}
    
    # We loop through all possible subject/session combinations from the start
    all_combinations = get_all_subject_combinations(config.PD_SX, config.CTL_SX)

    for subj_id, session_num, subj_type in all_combinations:
        processed_fname_base = f"{subj_id}_{session_num}_PD_REST_processed"
        processed_fname_full = os.path.join(config.PROCESSED_EEG_DATA_DIR, f"{processed_fname_base}{config.RAW_FNAME_SUFFIX}")
        
        if os.path.exists(processed_fname_full):
            try:
                epochs = mne.read_epochs(processed_fname_full, preload=True, verbose=False)
                
                # --- IMPORTANT FIX: Explicitly set subject info ---
                if 'subject_info' not in epochs.info or epochs.info['subject_info'] is None:
                    epochs.info['subject_info'] = {'id': subj_id}
                    
                all_loaded_epochs[(subj_id, subj_type)] = epochs
                print(f"Loaded {subj_type} data for subject {subj_id} session {session_num} with {len(epochs)} epochs.")
            except Exception as e:
                print(f"Error loading {processed_fname_full}: {e}. Skipping this file.")
        else:
            print(f"Warning: Processed file not found for {subj_type} subject {subj_id} session {session_num}. Skipping.")

    # --- Step 2: Perform Trial Matching on PD-CTL pairs ---
    # The new data structure: dict of lists, where each list element is a tuple of (PD_ON, PD_OFF, CTL) epochs
    matched_data = {'Eyes_Open': [], 'Eyes_Closed': []}
    
    # We loop through PD subjects from the `config` list to maintain pairing
    for pd_subj_id in config.PD_SX:
        try:
            # Look up epochs objects from the dictionary for the PD-CTL pair
            pd_on_epochs = all_loaded_epochs[(pd_subj_id, 'PD_ON')]
            pd_off_epochs = all_loaded_epochs[(pd_subj_id, 'PD_OFF')]
            
            # Find the corresponding CTL subject ID
            # Assuming CTLs and PDs are in the same order in the original Excel file
            ctl_subj_id = config.CTL_SX[config.PD_SX.index(pd_subj_id)]
            ctl_epochs = all_loaded_epochs[(ctl_subj_id, 'CTL')]
            
            # --- Trial Matching for 'Eyes_Open' ---
            count_pd_on_open = len(pd_on_epochs['Eyes_Open'])
            count_pd_off_open = len(pd_off_epochs['Eyes_Open'])
            count_ctl_open = len(ctl_epochs['Eyes_Open'])

            min_epochs_open = min(count_pd_on_open, count_pd_off_open, count_ctl_open)
            
            # Downsample to 100Hz before subsampling
            sub_pd_on_open = pd_on_epochs['Eyes_Open'].copy().resample(sfreq=config.FINAL_SAMPLING_RATE).pick(np.random.choice(count_pd_on_open, min_epochs_open, replace=False)).get_data()
            sub_pd_off_open = pd_off_epochs['Eyes_Open'].copy().resample(sfreq=config.FINAL_SAMPLING_RATE).pick(np.random.choice(count_pd_off_open, min_epochs_open, replace=False)).get_data()
            sub_ctl_open = ctl_epochs['Eyes_Open'].copy().resample(sfreq=config.FINAL_SAMPLING_RATE).pick(np.random.choice(count_ctl_open, min_epochs_open, replace=False)).get_data()
            
            matched_data['Eyes_Open'].append((sub_pd_on_open, sub_pd_off_open, sub_ctl_open))
            print(f"  Processed PD subject {pd_subj_id} / CTL {ctl_subj_id} 'Eyes_Open' with {min_epochs_open} epochs.")

            # --- Trial Matching for 'Eyes_Closed' ---
            count_pd_on_closed = len(pd_on_epochs['Eyes_Closed'])
            count_pd_off_closed = len(pd_off_epochs['Eyes_Closed'])
            count_ctl_closed = len(ctl_epochs['Eyes_Closed'])

            min_epochs_closed = min(count_pd_on_closed, count_pd_off_closed, count_ctl_closed)

            sub_pd_on_closed = pd_on_epochs['Eyes_Closed'].copy().resample(sfreq=config.FINAL_SAMPLING_RATE).pick(np.random.choice(count_pd_on_closed, min_epochs_closed, replace=False)).get_data()
            sub_pd_off_closed = pd_off_epochs['Eyes_Closed'].copy().resample(sfreq=config.FINAL_SAMPLING_RATE).pick(np.random.choice(count_pd_off_closed, min_epochs_closed, replace=False)).get_data()
            sub_ctl_closed = ctl_epochs['Eyes_Closed'].copy().resample(sfreq=config.FINAL_SAMPLING_RATE).pick(np.random.choice(count_ctl_closed, min_epochs_closed, replace=False)).get_data()

            matched_data['Eyes_Closed'].append((sub_pd_on_closed, sub_pd_off_closed, sub_ctl_closed))
            print(f"  Processed PD subject {pd_subj_id} / CTL {ctl_subj_id} 'Eyes_Closed' with {min_epochs_closed} epochs.")

        except KeyError:
            print(f"Warning: Processed file missing for PD subject {pd_subj_id} or its matched CTL. Skipping pair.")
            
    # --- Step 3: Save Aggregated and Matched Data ---
    print("\nSaving aggregated and trial-matched data...")
    with open(output_fname, 'wb') as f:
        pickle.dump(matched_data, f)
    print(f"Successfully saved to: {output_fname}")

# --- Run the main function ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR during data aggregation: {e}")
        import traceback
        traceback.print_exc()