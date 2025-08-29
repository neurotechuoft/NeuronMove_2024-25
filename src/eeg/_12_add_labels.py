import os
import pandas as pd
import numpy as np
import mne

class EEGLabeler:
    """
    A class to load EEG and accelerometer data, and add tremor labels to the EEG data.
    """
    def __init__(self, eeg_file_path, accel_file_path, eeg_sfreq=500, accel_sfreq=100):
        self.eeg_file_path = eeg_file_path
        self.accel_file_path = accel_file_path
        self.eeg_sfreq = eeg_sfreq
        self.accel_sfreq = accel_sfreq
        self.eeg_data = None
        self.accel_data = None
        self.upsampled_tremor_labels = None
        
    def load_data(self):
        """
        Loads the EEG and accelerometer data from their respective files.
        """
        print(f"Loading EEG data from: {os.path.basename(self.eeg_file_path)}")
        try:
            self.eeg_data = mne.io.read_raw_fif(self.eeg_file_path, preload=True, verbose=False)
            print("EEG data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: EEG file not found at {self.eeg_file_path}")
            self.eeg_data = None
            
        print(f"Loading accelerometer labels from: {os.path.basename(self.accel_file_path)}")
        try:
            self.accel_data = pd.read_pickle(self.accel_file_path)
            print("Accelerometer data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Accelerometer file not found at {self.accel_file_path}")
            self.accel_data = None

    def upsample_labels(self):
        """
        Upsamples the accelerometer tremor labels from 100 Hz to 500 Hz.
        """
        if self.accel_data is None or 'final_label' not in self.accel_data.columns:
            print("Error: Accelerometer data or 'tremor' column not available for upsampling.")
            return

        original_tremor_labels = self.accel_data['final_label'].values
        
        # Calculate the number of samples needed for upsampling
        resampling_factor = self.eeg_sfreq / self.accel_sfreq
        
        # Perform zero-order hold upsampling for the binary labels
        upsampled_labels = np.repeat(original_tremor_labels, resampling_factor)
        
        # Trim or pad the upsampled data to match the length of the EEG data
        eeg_num_samples = self.eeg_data.n_times
        if len(upsampled_labels) > eeg_num_samples:
            upsampled_labels = upsampled_labels[:eeg_num_samples]
        elif len(upsampled_labels) < eeg_num_samples:
            upsampled_labels = np.pad(upsampled_labels, (0, eeg_num_samples - len(upsampled_labels)), 'constant', constant_values=0)

        self.upsampled_tremor_labels = upsampled_labels.astype(int)
        print(f"Tremor labels upsampled to {self.eeg_sfreq} Hz.")

    def add_labels_to_eeg(self):
        """
        Adds the upsampled tremor labels as a channel to the EEG data.
        """
        if self.eeg_data is None or self.upsampled_tremor_labels is None:
            print("Error: EEG data or upsampled labels not available.")
            return

        info_tremor = mne.create_info(['TREMOR_LABEL'], self.eeg_sfreq, ['misc'])
        raw_tremor = mne.io.RawArray(self.upsampled_tremor_labels[np.newaxis, :], info_tremor, verbose=False)
        self.eeg_data.add_channels([raw_tremor], force_update_info=True)
        print("Tremor labels added to the EEG data.")

    def save_labeled_eeg(self, output_file_path):
        """
        Saves the EEG data with the new tremor label channel to a .fif file.
        """
        if self.eeg_data is None:
            print("Error: No labeled EEG data to save.")
            return

        self.eeg_data.save(output_file_path, overwrite=True, verbose=False)
        print(f"Labeled EEG data saved to: {os.path.basename(output_file_path)}")

def run_labeling_pipeline(eeg_dir, accel_dir, output_dir):
    """
    Main function to orchestrate the labeling pipeline for all files.
    """
    eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith('.fif')]
    accel_files = [f for f in os.listdir(accel_dir) if f.endswith('.pkl')]

    os.makedirs(output_dir, exist_ok=True)

    for eeg_file in eeg_files:
        # Extract the common identifier from the EEG filename
        eeg_base_name = eeg_file.split('_')[0] + '_' + eeg_file.split('_')[1]
        matching_accel_file = None
        for accel_file in accel_files:
            if eeg_base_name in accel_file:
                matching_accel_file = accel_file
                break

        if matching_accel_file:
            eeg_path = os.path.join(eeg_dir, eeg_file)
            accel_path = os.path.join(accel_dir, matching_accel_file)
            output_path = os.path.join(output_dir, f"labeled_{eeg_file}")
            
            labeler = EEGLabeler(eeg_path, accel_path)
            labeler.load_data()
            labeler.upsample_labels()
            labeler.add_labels_to_eeg()
            labeler.save_labeled_eeg(output_path)
        else:
            print(f"Warning: No matching accelerometer file found for {eeg_file}")

if __name__ == "__main__":
    eeg_data_dir = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/eeg/raw/"
    accel_data_dir = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/accelerometer/processed/"
    output_data_dir = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/eeg/labeled"

    run_labeling_pipeline(eeg_data_dir, accel_data_dir, output_data_dir)