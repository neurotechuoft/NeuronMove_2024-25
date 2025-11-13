import h5py
from pathlib import Path
import numpy as np

# --- Setup ---
base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / 'data' / 'raw' / 'parkinson_at_home' / 'sensor_data' / 'phys_cur_HC_merged.mat'

# Lists to store the sensor data from all participants
all_accel_data = []
all_gyro_data = []

print(f"Starting data extraction from: {file_path}")

try:
    with h5py.File(file_path, 'r') as f:
        # 1. Load the top-level 'phys' group
        phys_group = f['phys']
        
        # 2. Load the 'RW' (Right Wrist) data array of references
        rw_data_refs = phys_group['RW'][:] 
        
        num_participants = rw_data_refs.shape[0]
        print(f"\nFound {num_participants} participants to process in 'RW'.")

        # 3. Loop through each participant's reference
        for i in range(num_participants):
            # The reference is at [i, 0] in the (24, 1) array
            reference = rw_data_refs[i, 0]
            
            if isinstance(reference, h5py.Reference):
                # Dereference the object to get the actual data group
                participant_data_group = f[reference]
                
                # 4. Extract the 'accel' and 'gyro' datasets
                # .value is deprecated in newer h5py versions; use [:]
                
                # Accelerometer data
                accel_data = participant_data_group['accel'][:]
                all_accel_data.append(accel_data)
                
                # Gyroscope data
                gyro_data = participant_data_group['gyro'][:]
                all_gyro_data.append(gyro_data)
                
                print(f"Processed participant {i+1}/{num_participants}. Accel shape: {accel_data.shape}")
                
            else:
                print(f"Skipping participant {i+1}: Element is not a standard HDF5 reference.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

# --- Final Output Summary ---
if all_accel_data:
    print("\n--- Final Data Status ---")
    
    # Example of how you might combine all data into one single NumPy array
    # Note: If participants have different recording lengths, this will fail. 
    # It's safer to keep them as a list of arrays.
    # combined_accel = np.concatenate(all_accel_data, axis=0)
    
    print(f"Successfully extracted data from {len(all_accel_data)} participants.")
    print(f"The data is stored in the lists 'all_accel_data' and 'all_gyro_data'.")
    print(f"Example shape of the first participant's accel data: {all_accel_data[0].shape}")
    
    # The data is now ready for your analysis!

import pickle

# Define a path to save the processed data (e.g., in a 'processed' folder)
save_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw'/ 'parkinson_at_home' / 'sensor_data'
save_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

# Define the file path for the saved data
output_file_path = save_dir / 'phys_cur_HC_merged.pkl'

# Create a dictionary to hold all the data you want to save
data_to_save = {
    'accel': all_accel_data,
    'gyro': all_gyro_data
}

# Use pickle to save the dictionary
try:
    with open(output_file_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"\nData successfully saved to: {output_file_path}")
    
except Exception as e:
    print(f"\nERROR saving data: {e}")