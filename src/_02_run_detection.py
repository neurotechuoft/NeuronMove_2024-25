import os
from pathlib import Path
from src._01_tremor_detection import TremorDetector  

def run_pipeline_on_all_files(input_directory, output_directory):
    """
    Orchestrates the tremor detection pipeline on all files in a directory,
    EXCLUDING files located in the '/HC/' subdirectory.
    """
    # Find all accelerometer files in the input directory
    file_paths = TremorDetector.find_files(input_directory)
    
    # --- NEW FILTERING STEP: EXCLUDE HC FILES ---
    initial_count = len(file_paths)
    
    # Filter out any file path that contains the '/HC/' folder marker (for both Unix and Windows)
    filtered_file_paths = [
        p for p in file_paths 
        if '/HC/' not in str(p) and '\\HC\\' not in str(p)
    ]
    # --- END FILTERING STEP ---

    print(f"Found {initial_count} files total. Filtering to {len(filtered_file_paths)} non-HC files to process.")
    
    if not filtered_file_paths:
        print("No PD or other files found for processing. Exiting.")
        return

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    for file_path in filtered_file_paths:
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
            # Assuming you will tune this based on the previous discussion:
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
    base_dir = Path(__file__).parent.parent
    print(base_dir)

    # Input directory still points to the parent of HC/PD
    input_dir = base_dir / "data" / "raw" / "new_mexico" / "accelerometer"
    output_dir = base_dir / "data" / "raw" / "new_mexico" / "labeled"
    
    run_pipeline_on_all_files(input_dir, output_dir)
    print("\nAll files processed.")