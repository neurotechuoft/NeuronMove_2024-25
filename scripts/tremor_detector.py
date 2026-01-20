import os
import pandas as pd
import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.signal.windows import hamming
from spectrum import pburg
import subprocess
from pathlib import Path


class TremorDetector:
    """
    A class to handle data loading, preprocessing, and tremor detection for accelerometer data.
    """
    def __init__(self, data_path=None, fs=100.0, lowcut=1.0, highcut=30.0, fir_order=100, psd_order=6,
                 window_duration=3.0, overlap_ratio=0.90):
        self.data_path = data_path
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.fir_order = fir_order
        self.psd_order = psd_order
        self.window_duration = window_duration
        self.overlap_ratio = overlap_ratio
        
        self.raw_data = None
        self.data_no_drift = None
        self.filtered_data = None
        self.windowed_data = None
        self.psd_results = None
        self.df_features = None
        self.df_features_final = None
        self.upsampled_data = None
        self.df_smoothed_and_combined = None
        self.df_long_tremors = None

    def load_data(self):
        """
        Loads a single .pkl accelerometer file and prepares it for processing.
        """
        if self.data_path is None:
            print("Error: No data path specified.")
            return
        
        try:
            # Load the data, which is assumed to be a numpy array
            accel_data_array = pd.read_pickle(self.data_path)
            
            # Handle potential shape issues by transposing
            if accel_data_array.shape[0] < accel_data_array.shape[1]:
                accel_data_array = accel_data_array.T
            
            # Convert to DataFrame and store raw data
            self.raw_data = pd.DataFrame(accel_data_array, columns=['x', 'y', 'z'])

            self.raw_data = self.raw_data.astype(np.float64)
            
            print(f"Data from {os.path.basename(self.data_path)} loaded successfully.")
            print(f"Data type: {self.raw_data.dtypes}")

        except FileNotFoundError:
            print(f"Error: The file at {self.data_path} was not found.")
            self.raw_data = None
        
    def remove_drift(self, window_size=500):
        """
        Removes drift from the data using a moving average filter.
        
        Args:
            window_size (int): The window size for the moving average filter.
        """
        if self.raw_data is None:
            print("Error: Raw data is not available. Please load data first.")
            return

        # Calculate the moving average for each axis
        moving_average = self.raw_data.rolling(window=window_size, center=True).mean()

        # Subtract the moving average to remove drift and store the result
        self.data_no_drift = self.raw_data - moving_average

        # Drop the NaN values that result from the moving average calculation at the edges
        self.data_no_drift = self.data_no_drift.dropna()

        print("Drift removed from the data.")
        print("First 5 rows of the drift-removed data:")
        print(self.data_no_drift.head())

    def apply_fir_filter(self):
        """
        Applies a bandpass FIR filter to the drift-removed data.
        """
        if self.data_no_drift is None:
            print("Error: Drift-removed data is not available. Please run remove_drift first.")
            return
        
        b = firwin(self.fir_order + 1, [self.lowcut, self.highcut], pass_zero=False, fs=self.fs)
        y = filtfilt(b, 1.0, self.data_no_drift.values, axis=0)
        self.filtered_data = pd.DataFrame(y, columns=self.data_no_drift.columns)

        print(f"Data filtered successfully with a zero-phase FIR filter.")
        
    def apply_windowing(self, window_duration=3.0, overlap_ratio=0.90):
        """
        Applies a sliding Hamming window to the filtered data.
        """
        if self.filtered_data is None:
            print("Error: Filtered data is not available. Please run apply_fir_filter first.")
            return
        
        windowed_data = {}
        window_samples = int(window_duration * self.fs)
        overlap_samples = int(overlap_ratio * window_samples)
        step_samples = window_samples - overlap_samples
        ham_window = hamming(window_samples, sym=True)
        
        for axis in ['x', 'y', 'z']:
            data_series = self.filtered_data[axis].values
            windowed_segments = []
            for i in range(0, len(data_series) - window_samples + 1, step_samples):
                segment = data_series[i : i + window_samples]
                windowed_segment = segment * ham_window
                windowed_segments.append(windowed_segment)
            windowed_data[axis] = windowed_segments
            
        self.windowed_data = windowed_data
        segment = self.windowed_data['x'][0] if self.windowed_data['x'] else None
        print(f"Windowing completed for all axes. Number of windows: {len(self.windowed_data['x'])}")
        print("First 10 values of the segment:")
        print(segment[:10])
        print("\nStats of the segment:")
        print(f"Max value: {np.max(segment):.4f}")
        print(f"Min value: {np.min(segment):.4f}")
        print(f"Mean value: {np.mean(segment):.4f}")


    def _calculate_psd_berg_helper(self, segment):
        """
        Helper function to calculate PSD using the Berg method.
        FIX: Handle numerical precision errors and low-energy segments.
        """
        # Clean segment: ensure it's zero-mean
        segment_clean = segment - np.mean(segment)
        
        # Check if segment has enough energy to be meaningful
        segment_energy = np.sum(segment_clean ** 2)
        if segment_energy < 1e-10:
            # Segment is essentially flat/silent - return dummy spectrum
            n_freq = 256
            psd_vals = np.ones(n_freq) * 1e-20  # Very low power
            freqs = np.linspace(0, self.fs/2, n_freq)
            return psd_vals, freqs
        
        # Normalize to avoid numerical issues
        segment_normalized = segment_clean / (np.std(segment_clean) + 1e-10)
        
        try:
            from spectrum import pburg
            p = pburg(segment_normalized, order=self.psd_order, scale_by_freq=False)
            psd_vals = p.psd
            freqs = np.array(p.frequencies()) * self.fs
            
            # Ensure PSD values are positive (fix floating point errors)
            psd_vals = np.abs(psd_vals)
            
            # Denormalize back
            psd_vals = psd_vals * (np.std(segment_clean) ** 2 + 1e-10)
            
            return psd_vals, freqs
        except Exception as e:
            # If pburg fails, return very low power spectrum
            print(f"Warning: PSD calculation failed with error: {e}")
            print("Returning low-power spectrum")
            n_freq = 256
            psd_vals = np.ones(n_freq) * 1e-20
            freqs = np.linspace(0, self.fs/2, n_freq)
            return psd_vals, freqs

    def calculate_psd(self, order=6):
        """
        Calculates the Power Spectral Density (PSD) for each window using the Berg method.
        
        FIXES:
        1. Properly searches for peaks within 3-8 Hz range
        2. Handles numerical errors and low-energy segments
        3. Requires tremor peak to be dominant (80% of global max)
        4. Requires minimum absolute power threshold
        """
        if self.windowed_data is None:
            print("Error: Windowed data is not available. Please run apply_windowing first.")
            return
            
        psd_results = {}
        for axis in ['x', 'y', 'z']:
            psd_results[axis] = []
            for segment in self.windowed_data[axis]:
                # Calculate PSD for the segment
                psd_vals, freqs = self._calculate_psd_berg_helper(segment)
                
                # Find indices within the tremor frequency range (3-8 Hz)
                epsilon = 0.05
                tremor_range_mask = (freqs >= (3.0 - epsilon)) & (freqs <= (8.0 + epsilon))
                tremor_range_indices = np.where(tremor_range_mask)[0]
                
                # Find the global maximum
                global_max_idx = np.argmax(psd_vals)
                global_max_power = psd_vals[global_max_idx]
                global_max_freq = freqs[global_max_idx]
                
                # Minimum absolute power threshold to avoid detecting noise
                MIN_ABSOLUTE_POWER = 0.001
                
                # Check if there are any peaks in the tremor range
                if len(tremor_range_indices) > 0 and global_max_power > MIN_ABSOLUTE_POWER:
                    # Find the maximum peak power within the tremor range
                    tremor_range_psd = psd_vals[tremor_range_indices]
                    peak_power_index_in_range = np.argmax(tremor_range_psd)
                    actual_index = tremor_range_indices[peak_power_index_in_range]
                    
                    tremor_peak_power = psd_vals[actual_index]
                    tremor_peak_freq = freqs[actual_index]
                    
                    # STRICTER DOMINANCE CHECK:
                    # The tremor peak must be:
                    # 1. At least 80% of the global maximum (increased from 50%)
                    # 2. The global maximum must actually be in the tremor range
                    # This ensures we only detect tremor when it's truly the dominant frequency
                    
                    if global_max_freq >= 3.0 and global_max_freq <= 8.0:
                        # Global peak is in tremor range - this is likely tremor
                        peak_freq = tremor_peak_freq
                        peak_power = tremor_peak_power
                        label = 'tremor'
                    elif tremor_peak_power >= 0.8 * global_max_power:
                        # Tremor peak is very close to global peak
                        peak_freq = tremor_peak_freq
                        peak_power = tremor_peak_power
                        label = 'tremor'
                    else:
                        # There's a peak in tremor range, but it's not dominant
                        peak_freq = 1.0
                        peak_power = np.nan
                        label = 'non-tremor'
                else:
                    # No significant peak in tremor range
                    peak_freq = 1.0
                    peak_power = np.nan
                    label = 'non-tremor'
                
                psd_results[axis].append({
                    'psd': psd_vals,
                    'freqs': freqs,
                    'peak_freq': peak_freq,
                    'peak_power': peak_power,
                    'label': label
                })
        self.psd_results = psd_results
        print("PSD calculation completed for all windows.")


    def apply_power_threshold(self):
        """
        Applies a power threshold to refine the tremor labels.
        
        FIXES:
        1. Threshold is max/10 (as per paper)
        2. Better handling of edge cases
        3. Skip axes with insufficient tremor windows
        """
        if self.df_features is None:
            print("Error: Features DataFrame is not available. Please run create_features_dataframe first.")
            return

        # Create a copy to avoid modifying the original DataFrame
        df_features_final = self.df_features.copy()

        # Refine the labels based on the threshold
        df_features_final['final_label'] = 'non-tremor'

        for axis in ['x', 'y', 'z']:
            # Calculate the threshold for the current axis only
            axis_data = df_features_final[df_features_final['axis'] == axis]
            
            # Only calculate threshold from tremor-labeled windows
            tremor_data = axis_data[axis_data['label'] == 'tremor']
            
            if len(tremor_data) == 0:
                # No tremor detected in this axis at all
                continue
            
            # Remove NaN values before calculating max
            valid_powers = tremor_data['peak_power'].dropna()
            
            if len(valid_powers) == 0:
                # No valid power measurements
                continue
                
            max_peak_power = valid_powers.max()
            
            # Skip if max power is too small (likely noise)
            if max_peak_power < 0.01:
                print(f"Axis {axis}: max_power too low ({max_peak_power:.6f}), skipping")
                continue
                
            threshold = max_peak_power / 10  # CORRECTED: max/10 as per paper
            
            print(f"Axis {axis}: max_power={max_peak_power:.6f}, threshold={threshold:.6f}")
            
            # Apply the threshold to the current axis
            df_features_final.loc[
                (df_features_final['axis'] == axis) & 
                (df_features_final['label'] == 'tremor') & 
                (df_features_final['peak_power'] > threshold) &
                (df_features_final['peak_power'].notna()),  # Ensure not NaN
                'final_label'
            ] = 'tremor'

        # Set the frequency to 1 for all non-tremor windows
        df_features_final.loc[(df_features_final['final_label'] == 'non-tremor'), 'peak_freq_Hz'] = 1

        # Drop the intermediate columns
        df_features_final.drop(columns=['label', 'peak_power'], inplace=True)

        self.df_features_final = df_features_final
        
        tremor_count = len(self.df_features_final[self.df_features_final['final_label'] == 'tremor'])
        print(f"Power thresholding and final labeling complete.")
        print(f"Number of tremor windows detected: {tremor_count}")


    def create_features_dataframe(self):
        """
        Converts the psd_results into a pandas DataFrame.
        """
        if self.psd_results is None:
            print("Error: PSD results are not available. Please run calculate_psd first.")
            return

        data_list = []
        for axis in ['x', 'y', 'z']:
            for i, result in enumerate(self.psd_results[axis]):
                data_list.append({
                    'segment_id': i,
                    'axis': axis,
                    'peak_freq_Hz': result['peak_freq'],
                    'peak_power': result['peak_power'],
                    'label': result['label']
                })
        self.df_features = pd.DataFrame(data_list)
        print("Features DataFrame created successfully.")
        print("\nFirst 5 rows of the DataFrame:")
        print(self.df_features.head())
        print("\nLast 5 rows of the DataFrame:")
        print(self.df_features.tail())
        print("\nDataFrame Shape:", self.df_features.shape)


    def upsample_data(self):
        """
        Upsamples the data to match the original signal length.
        """
        if self.df_features_final is None:
            print("Error: Final features DataFrame is not available. Please run apply_power_threshold first.")
            return

        window_samples = int(self.window_duration * self.fs)
        step_samples = int(window_samples * (1 - self.overlap_ratio))
        total_samples = len(self.raw_data)

        upsampled_data = {}
        for axis in ['x', 'y', 'z']:
            df_axis = self.df_features_final[self.df_features_final['axis'] == axis].copy()
            original_time_points = df_axis['segment_id'].values * step_samples
            
            # Upsample the frequency data
            upsampled_data[f'{axis}_freq'] = np.interp(
                np.arange(total_samples), original_time_points, df_axis['peak_freq_Hz'].values, right=df_axis['peak_freq_Hz'].values[-1]
            )
            # Upsample the tremor labels
            upsampled_data[f'{axis}_tremor'] = np.interp(
                np.arange(total_samples), original_time_points, (df_axis['final_label'] == 'tremor').astype(int).values, right=0
            ).astype(int)

        self.upsampled_data = pd.DataFrame(upsampled_data)
        print("Data upsampled and stored in df_upsampled_data.")

    def smooth_and_combine_data(self, smoothing_window=300):
        """
        Smooths the upsampled frequency and combines the axes by multiplication.
        """
        if self.upsampled_data is None:
            print("Error: Upsampled data is not available. Please run upsample_data first.")
            return

        df_smoothed_freq = pd.DataFrame(index=self.upsampled_data.index)
        
        for col in self.upsampled_data.columns:
            df_smoothed_freq[col] = self.upsampled_data[col].rolling(
                window=smoothing_window, center=True).mean().fillna(0)

        combined_freq_signal = df_smoothed_freq['x_freq'] * df_smoothed_freq['y_freq'] * df_smoothed_freq['z_freq']
        df_smoothed_freq['combined'] = combined_freq_signal
        
        self.df_smoothed_and_combined = df_smoothed_freq
        print("Smoothed and combined data created successfully.")

    def extract_long_tremors(self, min_duration_s=3):
        """
        Extracts tremor events that persist for a minimum duration.
        """
        if self.df_smoothed_and_combined is None:
            print("Error: Smoothed and combined data is not available. Please run smooth_and_combine_data first.")
            return
            
        tremor_pulse = (self.df_smoothed_and_combined['combined'] > 3.5).astype(int)

        start_indices = np.where(np.diff(tremor_pulse, prepend=0) == 1)[0]
        end_indices = np.where(np.diff(tremor_pulse, append=0) == -1)[0]
        
        valid_tremor_events = []
        for start, end in zip(start_indices, end_indices):
            duration_in_seconds = (end - start) / self.fs
            if duration_in_seconds > (min_duration_s - 0.1):
                valid_tremor_events.append({
                    'start_sample': start,
                    'end_sample': end,
                    'duration_s': duration_in_seconds
                })
        
        self.df_long_tremors = pd.DataFrame(valid_tremor_events)
        print("Extracted long-duration tremor events successfully.")

    def add_tremor_labels_to_raw_data(self):
        """
        Adds a 'tremor' column to self.raw_data based on detected events.
        """
        if self.df_long_tremors is None:
            print("Error: Long tremors DataFrame is not available. Please run extract_long_tremors first.")
            return
            
        # Ensure raw_data exists and has an index
        if self.raw_data is None or self.raw_data.empty:
            print("Error: Raw data is not available or is empty.")
            return

        # Initialize the 'tremor' column to 0
        self.raw_data['tremor'] = 0

        # Iterate through the tremor events and set the corresponding samples to 1
        for _, row in self.df_long_tremors.iterrows():
            start_sample = int(row['start_sample'])
            end_sample = int(row['end_sample'])
            self.raw_data.loc[start_sample:end_sample, 'tremor'] = 1

        print("Added 'tremor' labels to raw_data DataFrame successfully.")

    @staticmethod
    def find_files(directory):
        """
        Finds all .pkl files in the given directory.
        
        Args:
            directory (str): The path to the directory.
            
        Returns:
            list: A list of full file paths.
        """
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.pkl'):
                    file_paths.append(os.path.join(root, file))
        return file_paths
    

if __name__ == "__main__":
    # Define the path to the specific file you want to test
    base_dir = Path(__file__).parent.parent

    specific_file_path = base_dir / "data" / "raw" / "new_mexico" / "accelerometer"/ "801_1_accelerometer.pkl"
    
    # Instantiate the TremorDetector class
    detector = TremorDetector(data_path=specific_file_path)
    
    # Load the data
    detector.load_data()
    
    # Check if data was loaded successfully before proceeding
    if detector.raw_data is not None:
        # Run the drift removal method
        detector.remove_drift()
        
        if detector.data_no_drift is not None:
            detector.apply_fir_filter()
            
            if detector.filtered_data is not None:
                detector.apply_windowing()
                
                if detector.windowed_data is not None:
                    detector.calculate_psd()
                    
                    if detector.psd_results is not None:
                        detector.create_features_dataframe()

                        if detector.df_features is not None:
                            detector.apply_power_threshold()
                            
                            if detector.df_features_final is not None:
                                detector.upsample_data()
                                
                                if detector.upsampled_data is not None:
                                    detector.smooth_and_combine_data()
                                    
                                    if detector.df_smoothed_and_combined is not None:
                                        detector.extract_long_tremors()
                                        
                                        if detector.df_long_tremors is not None:
                                            detector.add_tremor_labels_to_raw_data()
                                            
                                            # Convert the DataFrame to a string
                                            df_string = detector.raw_data.to_string()

                                            # Use 'less' to display the string in the terminal
                                            p = subprocess.Popen(['less'], stdin=subprocess.PIPE)
                                            p.stdin.write(df_string.encode('utf-8'))
                                            p.stdin.close()
                                            p.wait()