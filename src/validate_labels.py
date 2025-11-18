import pandas as pd
from pathlib import Path
import os

def analyze_tremor_counts(base_dir: Path):
    """
    Loads all labeled PD and HC files based on the specified structure, 
    calculates the total tremor samples for each, and saves the results to a CSV file.

    Args:
        base_dir (Path): The root directory of the project.
    """
    
    # --- 1. Define the labeled data root path and specific patient type paths ---
    labeled_data_root = base_dir / "data" / "raw" / "new_mexico" / "labeled"
    
    # Define the explicit paths based on your structure:
    # PD files are directly in 'labeled/'; HC files are in 'labeled/HC/'
    target_paths = {
        "PD": labeled_data_root,           
        "HC": labeled_data_root / "HC"     
    }
    
    results = []

    print(f"Starting tremor analysis in {labeled_data_root}...")

    # --- 2. Iterate through PD and HC target paths ---
    for patient_type, target_dir in target_paths.items():
        
        if not target_dir.is_dir():
            print(f"Warning: Directory not found: {target_dir}. Skipping.")
            continue
            
        print(f"\nSearching for {patient_type} files in: {target_dir}")

        # MODIFIED: Search for ALL .pkl files in the PD directory for flexibility
        # For HC, we keep the specific *_labeled.pkl pattern for clarity
        glob_pattern = "*.pkl" if patient_type == "PD" else "*_labeled.pkl"

        for input_file in target_dir.glob(glob_pattern):
            
            # Skip the HC directory itself if it is found while globbing PD files
            if input_file.is_dir():
                continue

            # Skip already labeled HC files if they were found accidentally
            if patient_type == "HC" and "_labeled.pkl" not in input_file.name:
                continue
                
            try:
                # Load the labeled data (expected to be a DataFrame)
                df_labeled = pd.read_pickle(input_file)
                
                if 'tremor' not in df_labeled.columns:
                    print(f"Error: 'tremor' column not found in {input_file.name}. Skipping.")
                    continue

                total_tremor_samples = df_labeled['tremor'].sum()
                total_samples = len(df_labeled)

                # MODIFIED: Get Subject ID based on file type
                if patient_type == "HC":
                    # e.g., "901_1_labeled.pkl" -> "901_1"
                    subject_id = input_file.stem.split('_labeled')[0]
                else:
                    # For PD, use the whole file stem (e.g., "PD_data_001.pkl" -> "PD_data_001")
                    subject_id = input_file.stem

                results.append({
                    'Subject_ID': subject_id,
                    'Type': patient_type,
                    'Total_Samples': total_samples,
                    'Tremor_Samples': total_tremor_samples,
                    'Tremor_Duration_s': total_tremor_samples / 100.0, # Assuming fs=100Hz
                })

            except Exception as e:
                print(f"Error processing file {input_file.name}: {e}")
                continue

    # --- 3. Save and Display Results (Unchanged) ---
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by=['Type', 'Tremor_Samples'], ascending=[True, False])

        # Define the output path for the CSV file
        csv_output_path = base_dir / "data" / "raw" / "new_mexico" / "tremor_analysis_summary.csv"
        
        # Save the DataFrame to CSV
        df_results.to_csv(csv_output_path, index=False)
        print("\n" + "="*50)
        print(f"Summary saved successfully to: {csv_output_path}")
        print("="*50)

        # Display the results table in the console
        print("\n" + "="*50)
        print("                 TREMOR ANALYSIS SUMMARY                ")
        print("="*50)
        
        try:
             print(df_results.to_markdown(index=False, floatfmt=".2f"))
        except ImportError:
            print(df_results)
        
        summary_by_type = df_results.groupby('Type')['Tremor_Duration_s'].agg(['sum', 'mean', 'count']).reset_index()
        summary_by_type.columns = ['Type', 'Total_Tremor_Time (s)', 'Avg_Tremor_Time (s)', 'Count']
        
        print("\nGroup Summary:")
        try:
            print(summary_by_type.to_markdown(index=False, floatfmt=".2f"))
        except ImportError:
            print(summary_by_type)

    else:
        print("No labeled files were found or processed.")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    analyze_tremor_counts(base_dir)