# NTUT25_SOFTWARE/src/eeg/01_preprocess_eeg.py

import os
import pandas as pd
from . import config 
from ._11_eeg_preprocessor import EEGProcessor 

def main():
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Raw EEG data directory: {config.RAW_EEG_DATA_DIR}")
    print(f"Processed EEG data directory: {config.PROCESSED_EEG_DATA_DIR}")

    os.makedirs(config.PROCESSED_EEG_DATA_DIR, exist_ok=True)

    try:
        demographics_df = pd.read_excel(os.path.join(config.MISC_DATA_DIR, 'IMPORT_ME.xlsx'))
    except FileNotFoundError:
        print(f"Warning: IMPORT_ME.xlsx not found at {os.path.join(config.MISC_DATA_DIR, 'IMPORT_ME.xlsx')}.")
        print("Subject type checks (PD/CTL) will rely solely on hardcoded lists.")
        demographics_df = None

    for subj_id in config.ALL_SUBJECTS:
        is_pd = subj_id in config.PD_SX
        is_ctl = subj_id in config.CTL_SX

        sessions_to_process = []
        if is_pd:
            sessions_to_process = [1, 2]
        elif is_ctl:
            sessions_to_process = [1]
        
        for session_num in sessions_to_process:
            processor = EEGProcessor(subj_id, session_num, config)
            processor.process_subject()

    print("\n--- All Preprocessing Runs Attempted ---")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR during overall pipeline execution: {e}")
        import traceback
        traceback.print_exc()