import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config as cfg

def load_data(file_path: Path) -> None:
    """
    Load raw pickel data file.
    """
    print(f"[Loading Mode] Attempting to load data from: {file_path}") # Debug print
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print("Data successfully loaded.")
            return data
    except FileNotFoundError:
        print(f"\nERROR: The file was not found at {file_path}") # Debug print
        return None
    except Exception as e:
        print(f"\nERROR loading data: Could not unpickle the file. {e}") # Debug print
        return None

def visualize_data(data: dict[str, list[np.ndarray]]) -> None:
    """
    Creates a two-panel visualization for the first participant's accelerometer and gyroscope data.
    """
    if not data or 'accel' not in data or not data['accel']:
        print("No valid sensor data available for visualization.") # Debug print
        return

    accel = data['accel'][0] # Get the data for the first participant (index 0)
    gyro = data['gyro'][0]

    accel_T = accel.T   # Transpose the data from (4, N) to (N, 4) for plotting
    gyro_T = gyro.T     # N=samples, 4=Time/Index + 3 axes (X, Y, Z)
    
    if accel_T.shape[1] >= 4 and gyro_T.shape[1] >= 4:      # Check for expected shape (at least 4 columns: Time/Index, X, Y, Z)

        accel_len = accel_T.shape[0]        # Ensure Accelerometer and Gyroscope arrays have the same length -- taking both from index 0
        gyro_len = gyro_T.shape[0]
        
        min_samples = min(accel_len, gyro_len)
        accel_T = accel_T[:min_samples, :]
        gyro_T = gyro_T[:min_samples, :]         # Truncate both arrays to the minimum length
        
        time_index = np.arange(min_samples)          # Generate the time index based on the unified length

        print(f"Visualization: Truncating data to {min_samples} samples for matching dimensions.") # Debug print
        
        # --- Visualization ---
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        fig.suptitle(f"Participant 1 Right Wrist Sensor Data (Total Samples: {min_samples})", fontsize=16)

        # 1. Accelerometer Plot (Columns 1, 2, 3 are X, Y, Z data)
        axes[0].plot(time_index, accel_T[:, 1], label='X-Axis', linewidth=1)
        axes[0].plot(time_index, accel_T[:, 2], label='Y-Axis', linewidth=1)
        axes[0].plot(time_index, accel_T[:, 3], label='Z-Axis', linewidth=1)
        axes[0].set_title('Right Wrist Accelerometer Data (m/s²)')
        axes[0].set_ylabel('Acceleration (a.u.)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. Gyroscope Plot (Columns 1, 2, 3 are X, Y, Z data)
        axes[1].plot(time_index, gyro_T[:, 1], label='X-Axis', linewidth=1)
        axes[1].plot(time_index, gyro_T[:, 2], label='Y-Axis', linewidth=1)
        axes[1].plot(time_index, gyro_T[:, 3], label='Z-Axis', linewidth=1)
        axes[1].set_title('Right Wrist Gyroscope Data (rad/s)')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Angular Velocity (a.u.)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
        plt.show()
    else:
        print("Data shape is unexpected for plotting (expected at least 4 columns: Index, X, Y, Z).")

if __name__ == "__main__":
    file_names = [
        'phys_cur_HC_merged.pkl',
        'phys_cur_PD_merged.pkl'
    ]

    for file_name in file_names:
        file_path = cfg.SENSOR_DATA / file_name
        sensor_data = load_data(file_path)

        if sensor_data is not None:
            visualize_data(sensor_data)