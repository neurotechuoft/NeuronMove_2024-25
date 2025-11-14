import sys
from pathlib import Path
import pandas as pd # Essential for DataFrame operations
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# --- Module Setup ---
# Add project root to the Python path to allow importing 'config'
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as cfg

# --- Configuration ---
ANNOTATION_DIR = cfg.ANNOTATION_DATA 
SENSOR_DATA_DIR = cfg.SENSOR_DATA # Directory for the sensor PKL files
SAMPLING_FREQUENCY = 200 # Hz, as per the dataset description

def load_data(pkl_file_path: Path) -> Dict[str, Any] | None:
    """Safely loads data from a pickle file."""
    if not pkl_file_path.exists():
        return None
    try:
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_file_path.name}: {e}")
        return None

def unwrap_value(series_or_value: Any) -> Any:
    """Helper function to safely extract the value from nested array/list structures."""
    if isinstance(series_or_value, np.ndarray) and series_or_value.ndim > 0:
        # Check if the array is just a wrapper for content
        if series_or_value.size == 1:
            return unwrap_value(series_or_value.flat[0])
        return series_or_value
    
    # Handle list of lists (common MATLAB output for empty values)
    if isinstance(series_or_value, list):
        if len(series_or_value) == 1:
            return unwrap_value(series_or_value[0])
        return series_or_value # Return list if complex
        
    return series_or_value

def visualize_tremor_epochs(sensor_data_dict: Dict[str, List[np.ndarray]], labels_data_dict: Dict[str, Any], participant_index: int):
    """
    Plots the sensor data for a participant and highlights the annotated tremor periods.
    """
    
    # 1. Prepare Sensor Data for the Target Participant (Patient 2 is index 1)
    try:
        accel_raw = sensor_data_dict['accel'][participant_index]
        gyro_raw = sensor_data_dict['gyro'][participant_index]
        
        # Transpose and truncate to minimum length for plotting compatibility
        accel_T = accel_raw.T
        gyro_T = gyro_raw.T
        
        min_samples = min(accel_T.shape[0], gyro_T.shape[0])
        accel_T = accel_T[:min_samples]
        gyro_T = gyro_T[:min_samples]
        
        time_index = np.arange(min_samples) / SAMPLING_FREQUENCY # Time in seconds
        
    except IndexError:
        print(f"ERROR: Participant index {participant_index} out of bounds for sensor data.")
        return

    # 2. Prepare Annotation Data for the Target Participant
    try:
        # The 'labels' key holds the structured NumPy array
        labels_array = labels_data_dict['labels']
        
        # Convert to DataFrame to easily access named columns
        df_labels = pd.DataFrame(labels_array)
        
        # Patient 2 is the second row in the DataFrame (index 1)
        patient_row = df_labels.iloc[participant_index]
        
        # Get Patient ID for Title
        raw_id = unwrap_value(patient_row['id'])
        
        # Safely convert raw ID to string
        if isinstance(raw_id, (bytes, np.bytes_)):
            patient_id = raw_id.decode('utf-8')
        else:
            patient_id = str(raw_id)
        
        # --- Extract Tremor Timestamps (PRE-medication) ---
        # NOTE: Changed from 'postmed_' to 'premed_'
        start_raw = unwrap_value(patient_row['premed_tremorstart'])
        end_raw = unwrap_value(patient_row['premed_tremorend'])

        # Flatten and convert to a list of floats (if not empty)
        tremor_starts_s = np.array(start_raw).flatten().tolist() if start_raw is not None and np.array(start_raw).size > 0 else []
        tremor_ends_s = np.array(end_raw).flatten().tolist() if end_raw is not None and np.array(end_raw).size > 0 else []

        if not tremor_starts_s:
             print(f"NOTE: No 'premed' tremor epochs found for {patient_id}. Showing full time series.")
        
    except Exception as e:
        print(f"ERROR processing annotation data for patient {participant_index}: {e}")
        return

    # 3. Create Visualization
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(f"Sensor Data and Annotated Tremor Epochs for Patient {patient_id} (Pre-Medication)", fontsize=16)

    # --- Plot 1: Accelerometer ---
    axes[0].plot(time_index, accel_T[:, 1], label='Accel X', linewidth=1, alpha=0.7)
    axes[0].plot(time_index, accel_T[:, 2], label='Accel Y', linewidth=1, alpha=0.7)
    axes[0].plot(time_index, accel_T[:, 3], label='Accel Z', linewidth=1, alpha=0.7)
    axes[0].set_title('Right Wrist Accelerometer Data (g)')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].legend(loc='upper right')

    # --- Plot 2: Gyroscope ---
    axes[1].plot(time_index, gyro_T[:, 1], label='Gyro X', linewidth=1, alpha=0.7)
    axes[1].plot(time_index, gyro_T[:, 2], label='Gyro Y', linewidth=1, alpha=0.7)
    axes[1].plot(time_index, gyro_T[:, 3], label='Gyro Z', linewidth=1, alpha=0.7)
    axes[1].set_title('Right Wrist Gyroscope Data (rad/s)')
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].legend(loc='upper right')

    # 4. Highlight Tremor Epochs on Both Plots (Shading and Vertical Lines)
    is_first_epoch = True
    for start_s, end_s in zip(tremor_starts_s, tremor_ends_s):
        # We assume the timestamps are in seconds relative to the start of the recording
        
        label_text = 'Tremor Period' if is_first_epoch else None

        for ax in axes:
            # Shaded Area (axvspan)
            ax.axvspan(start_s, end_s, color='red', alpha=0.2, label=label_text)
            
            # Vertical Lines (axvline) for Start/End
            ax.axvline(start_s, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(end_s, color='red', linestyle='--', linewidth=1.5)
        
        is_first_epoch = False
            
    # Remove duplicate legend entries if multiple tremor epochs were shaded
    if tremor_starts_s:
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc='upper right')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit suptitle
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    
    # Define file paths for the PD data and annotations
    sensor_pkl_file = SENSOR_DATA_DIR / 'phys_cur_PD_merged.pkl'
    labels_pkl_file = ANNOTATION_DIR / 'labels_PD_phys_tremor.pkl'
    
    # Load the data
    sensor_data_dict = load_data(sensor_pkl_file)
    labels_data_dict = load_data(labels_pkl_file)
    
    if sensor_data_dict is None or labels_data_dict is None:
        print("\nFATAL ERROR: Could not load one or both required PKL files.")
        print("Ensure you have run src/load_data.py and src/extract_annotations.py.")
    else:
        # Visualize the data for Patient 2 (index 1), who is known to have tremor data
        visualize_tremor_epochs(sensor_data_dict, labels_data_dict, participant_index=1)