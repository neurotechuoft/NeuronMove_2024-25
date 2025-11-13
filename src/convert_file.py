import h5py # MATLAB v7.3 files are HDF5 format
from pathlib import Path
import numpy as np
import config as cfg
import pickle

file_path = cfg.SENSOR_DATA / 'phys_cur_HC_merged.mat'

all_accel_data = []
all_gyro_data = []

print(f"Starting data extraction from: {file_path}") 

try:
    with h5py.File(file_path, 'r') as f:
        phys_group = f['phys']  # print(list(f.keys()))
        rw_data_refs = phys_group['RW'][:] # type: ignore
        num_participants = rw_data_refs.shape[0] # type: ignore
        print(f"\nFound {num_participants} participants to process in 'RW'.")

        for i in range(num_participants):
            reference = rw_data_refs[i, 0]  # type: ignore 
            # The reference is at [i, 0] in the (24, 1) array rw_data_refs.shape (24 participants, sensor data stacked in column)

            if isinstance(reference, h5py.Reference):
                participant_data_group = f[reference] # Dereference the object to get the actual data group
                accel_data = participant_data_group['accel'][:] # type: ignore
                all_accel_data.append(accel_data)
                gyro_data = participant_data_group['gyro'][:] # type: ignore
                all_gyro_data.append(gyro_data)
                print(f"Processed participant {i+1}/{num_participants}. Accel shape: {accel_data.shape}") # type: ignore
            else:
                print(f"Skipping participant {i+1}: Element is not a standard HDF5 reference.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

if all_accel_data:
    print("\n--- Final Data Status ---")
    # If participants have different recording lengths, this will fail. 
    # It's safer to keep them as a list of arrays.
    # combined_accel = np.concatenate(all_accel_data, axis=0)
    print(f"Successfully extracted data from {len(all_accel_data)} participants.")
    print(f"The data is stored in the lists 'all_accel_data' and 'all_gyro_data'.")
    print(f"Example shape of the first participant's accel data: {all_accel_data[0].shape}")

# save_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist
output_file_path = cfg.SENSOR_DATA / 'phys_cur_HC_merged.pkl'

# Dictionary: two keys 'accel' and 'gyro', each mapping to a list of numpy arrays (one per participant)
data_to_save = {
    'accel': all_accel_data,
    'gyro': all_gyro_data
}

# Save the dictionary (pickle)
try:
    with open(output_file_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"\nData successfully saved to: {output_file_path}")
    
except Exception as e:
    print(f"\nERROR saving data: {e}")