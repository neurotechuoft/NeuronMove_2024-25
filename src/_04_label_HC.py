import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def label_healthy_control(input_path: str):
    """
    Loads raw accelerometer data (NumPy array format), converts it to a DataFrame, 
    adds a 'tremor' column initialized to 0 for all samples, and saves the result 
    to a new .pkl file.

    Args:
        input_path: The path to the raw input .pkl file containing the NumPy array.
    """
    input_file = Path(input_path)
    print(f"Loading raw data from: {input_file}")

    try:
        accel_data_array = pd.read_pickle(input_file)
        
        if accel_data_array.shape[0] < accel_data_array.shape[1]:
            accel_data_array = accel_data_array.T
        
        df_raw = pd.DataFrame(accel_data_array, columns=['x', 'y', 'z'])
        df_raw = df_raw.astype(np.float64)
        
        print(f"Data loaded and converted to DataFrame with {len(df_raw)} samples.")

    except FileNotFoundError:
        print(f"Error: The file at {input_file} was not found.")
        return
    except Exception as e:
        print(f"Error processing pickle file: {e}")
        return

    df_raw['tremor'] = 0
    print("All samples successfully labeled as non-tremor (tremor=0).")

    output_dir = base_dir / "data" / "raw" / "new_mexico" / "labeled" / "HC"

    output_filename = input_file.stem.replace('_accelerometer', '') + '_labeled.pkl'

    output_path = output_dir / output_filename

    output_dir.mkdir(parents=True, exist_ok=True) 
    print(f"Ensured output directory exists: {output_dir}")
    try:
        df_raw.to_pickle(output_path)
        print(f"Labeled data saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent

    hc_data_dir = base_dir / "data" / "raw" / "new_mexico" / "accelerometer" / "HC"
    
    print(f"Starting labeling process in: {hc_data_dir}")

    for input_file in hc_data_dir.glob("*.pkl"):
        if "_labeled.pkl" in input_file.name:
            print(f"Skipping already labeled file: {input_file.name}")
            continue

        print(f"\n--- Processing file: {input_file.name} ---")
        
        label_healthy_control(str(input_file)) 
        
    print("\nLabeling of all Healthy Control files complete.")