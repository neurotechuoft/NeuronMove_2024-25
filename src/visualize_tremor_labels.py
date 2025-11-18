import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def visualize_tremor_labels(base_dir: Path):
    """
    Loads all labeled PD and HC files and creates a time-series plot of 
    the raw data with detected tremor windows highlighted in red.
    """
    
    # --- 1. Define Directories and Target Paths ---
    labeled_data_root = base_dir / "data" / "raw" / "new_mexico" / "labeled"
    vis_output_dir = labeled_data_root / "visualizations"
    
    # Define the explicit paths based on your structure
    target_paths = {
        "PD": labeled_data_root,           
        "HC": labeled_data_root / "HC"     
    }
    
    # Ensure the output directory for visualizations exists at the very start
    try:
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {vis_output_dir}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not create output directory {vis_output_dir}. Check permissions. Error: {e}")
        return

    # --- 2. Iterate through PD and HC files ---
    for patient_type, target_dir in target_paths.items():
        
        if not target_dir.is_dir():
            print(f"Warning: Directory not found: {target_dir}. Skipping {patient_type}.")
            continue
            
        print(f"\nProcessing {patient_type} files in: {target_dir}")
        
        # Define specific glob patterns for each file type
        glob_pattern = "tremor_labels_*_accelerometer.pkl" if patient_type == "PD" else "*_labeled.pkl"

        for input_file in target_dir.glob(glob_pattern):
            
            # Skip directories and files that might be the original raw files
            if input_file.is_dir() or "_accelerometer.pkl" in input_file.name and patient_type == "HC":
                continue

            # --- Subject ID Extraction (Used for Title/Filename) ---
            if patient_type == "PD":
                subject_id = input_file.stem.replace('tremor_labels_', '').replace('_accelerometer', '')
            else:
                subject_id = input_file.stem.split('_labeled')[0]
            
            try:
                # Load the labeled data
                df = pd.read_pickle(input_file)
                
                if 'tremor' not in df.columns:
                    print(f"  - Skipping {input_file.name}: 'tremor' column missing.")
                    continue
                
                # --- 3. Plotting and Red Highlighting Logic ---
                
                # Assume 100 Hz sampling rate for time conversion
                time = df.index.values / 100.0  
                plot_axis = 'y' 
                
                plt.figure(figsize=(18, 5))
                plt.plot(time, df[plot_axis], label=f'{plot_axis.upper()}-Axis Acceleration', color='C0', linewidth=0.5)
                
                # Identify the start and end of continuous tremor periods
                tremor_samples = df['tremor'].values
                tremor_start_indices = np.where(np.diff(tremor_samples, prepend=0) == 1)[0]
                tremor_end_indices = np.where(np.diff(tremor_samples, append=0) == -1)[0]
                
                tremor_starts = tremor_start_indices / 100.0
                tremor_ends = tremor_end_indices / 100.0
                
                # Apply the shaded background (red highlight)
                highlight_label_added = False
                for start, end in zip(tremor_starts, tremor_ends):
                    label = 'Tremor Detected' if not highlight_label_added else None
                    plt.axvspan(start, end, color='red', alpha=0.3, label=label)
                    highlight_label_added = True
                
                # --- Final Plot Aesthetics ---
                handles, labels = plt.gca().get_legend_handles_labels()
                unique_labels = dict(zip(labels, handles))
                
                plt.title(f"Tremor Detection Visualization: Subject {subject_id} ({patient_type})")
                plt.xlabel("Time (s)")
                plt.ylabel(f"{plot_axis.upper()}-Axis Accel. (g)")
                plt.legend(unique_labels.values(), unique_labels.keys())
                plt.grid(True, alpha=0.5)
                
                # --- 4. Debug and Save the figure ---
                output_filename = f"{subject_id}_tremor_vis.png"
                output_path = vis_output_dir / output_filename
                
                # --- DEBUG CHECK 1: Confirm Save Path ---
                print(f"  - Attempting to save to: {output_path}")

                try:
                    plt.savefig(output_path)
                    plt.close() # Close the figure immediately after saving
                    print(f"  - SUCCESS: Visualization saved for {subject_id}")
                except Exception as save_e:
                    # --- DEBUG CHECK 2: CRITICAL SAVE ERROR ---
                    print(f"  - CRITICAL SAVE ERROR for {subject_id}: {save_e}")
                    plt.close()

            except Exception as e:
                print(f"  - Error processing {input_file.name}: {e}")
                continue

    print("\nVisualization complete. Check the 'visualizations' folder for output files.")

if __name__ == "__main__":
    # Define the project root
    base_dir = Path(__file__).parent.parent
    visualize_tremor_labels(base_dir)