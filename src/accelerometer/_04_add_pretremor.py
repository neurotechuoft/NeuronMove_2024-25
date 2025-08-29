import os
import pandas as pd
import numpy as np

class PreTremorLabeler:
    """
    A class to add pre-tremor labels to accelerometer data.
    """
    def __init__(self, raw_data_with_labels):
        self.raw_data = raw_data_with_labels
        self.labeled_data = None

    def add_pre_tremor_labels(self, pre_tremor_duration_s=3):
        """
        Adds a 'pre-tremor' column to the raw data based on tremor onset.
        
        Args:
            pre_tremor_duration_s (int): The duration of the pre-tremor segment in seconds.
        """
        if self.raw_data is None or 'tremor' not in self.raw_data.columns:
            print("Error: DataFrame is missing 'tremor' column or is empty.")
            return

        self.labeled_data = self.raw_data.copy()
        
        # 0 = non-tremor, 2 = tremor
        self.labeled_data['final_label'] = self.labeled_data['tremor'].replace({0: 0, 1: 2})
        
        tremor_onsets = np.where(np.diff(self.labeled_data['final_label'], prepend=0) == 2)[0]
        
        fs = 100
        pre_tremor_samples = int(pre_tremor_duration_s * fs)
        
        for onset_index in tremor_onsets:
            pre_tremor_start = onset_index - pre_tremor_samples
            pre_tremor_end = onset_index - 1
            
            if pre_tremor_start < 0:
                pre_tremor_start = 0
            
            self.labeled_data.loc[
                (self.labeled_data.index >= pre_tremor_start) &
                (self.labeled_data.index <= pre_tremor_end) &
                (self.labeled_data['final_label'] == 0),
                'final_label'
            ] = 1

        # --- Corrected: Drop the old 'tremor' column ---
        self.labeled_data.drop(columns=['tremor'], inplace=True)
        # --- End of corrected line ---

        print("Pre-tremor labels added successfully. Old 'tremor' column removed.")
        
    @staticmethod
    def find_files(directory, suffix=".pkl"):
        """
        Finds all files with a specific suffix in the given directory.
        """
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(suffix):
                    file_paths.append(os.path.join(root, file))
        return file_paths

def run_pre_tremor_labeling(input_dir, output_dir):
    """
    Main function to run the pre-tremor labeling pipeline on all files.
    """
    file_paths = PreTremorLabeler.find_files(input_dir, suffix=".pkl")
    if not file_paths:
        print("No files found. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in file_paths:
        try:
            print(f"Processing file: {os.path.basename(file_path)}")
            
            data_with_labels = pd.read_pickle(file_path)
            
            labeler = PreTremorLabeler(raw_data_with_labels=data_with_labels)
            labeler.add_pre_tremor_labels()
            
            output_filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, output_filename)
            labeler.labeled_data.to_pickle(output_path)
            
            print(f"Successfully saved final labels to {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            
if __name__ == "__main__":
    input_dir = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/accelerometer/processed/"
    output_dir = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/accelerometer/processed/"
    
    run_pre_tremor_labeling(input_dir, output_dir)