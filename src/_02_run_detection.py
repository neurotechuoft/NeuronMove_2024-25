import os
from src.accelerometer._02_tremor_labeling import TremorDetector  

def run_pipeline_on_all_files(input_directory, output_directory):
    """
    Orchestrates the tremor detection pipeline on all files in a directory.
    """
    # Find all accelerometer files in the input directory
    file_paths = TremorDetector.find_files(input_directory)
    print(f"Found {len(file_paths)} files to process.")
    
    if not file_paths:
        print("No files found. Exiting.")
        return

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    for file_path in file_paths:
        try:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            
            # Instantiate the detector with the current file
            detector = TremorDetector(data_path=file_path)
            
            # Run the complete pipeline
            detector.load_data()
            detector.remove_drift()
            detector.apply_fir_filter()
            detector.apply_windowing()
            detector.calculate_psd()
            detector.create_features_dataframe()
            detector.apply_power_threshold()
            detector.upsample_data()
            detector.smooth_and_combine_data()
            detector.extract_long_tremors()
            detector.add_tremor_labels_to_raw_data()
            
            # Save the final output to the output directory
            if detector.raw_data is not None:
                output_filename = f"tremor_labels_{os.path.basename(file_path).split('.')[0]}.pkl"
                output_path = os.path.join(output_directory, output_filename)
                
                detector.raw_data.to_pickle(output_path)
                print(f"Successfully saved tremor labels to {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    # --- Configuration ---
    input_dir = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/accelerometer/raw/"
    output_dir = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/accelerometer/processed/"
    
    run_pipeline_on_all_files(input_dir, output_dir)
    print("\nAll files processed.")