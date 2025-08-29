import os
import pandas as pd
import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.signal.windows import hamming
from spectrum import pburg
from IPython.display import display, HTML
import subprocess
import io


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

    def calculate_psd(self, order=6):
        """
        Calculates the Power Spectral Density (PSD) for each window using the Berg method.
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
                
                # Find the global peak power and its corresponding frequency in the entire spectrum
                peak_power_index = np.argmax(psd_vals)
                global_peak_freq = freqs[peak_power_index]
                global_peak_power = psd_vals[peak_power_index]
                
                label = 'non-tremor'
                if 3.0 <= global_peak_freq <= 8.0:
                    peak_freq = global_peak_freq
                    peak_power = global_peak_power
                    label = 'tremor'
                else:
                    peak_freq = 1.0  # According to the paper
                    peak_power = np.nan
                
                psd_results[axis].append({
                    'psd': psd_vals,
                    'freqs': freqs,
                    'peak_freq': peak_freq,
                    'peak_power': peak_power,
                    'label': label
                })
        self.psd_results = psd_results
        print("PSD calculation completed for all windows.")

    def _calculate_psd_berg_helper(self, segment):
        """Helper function to calculate PSD using the Berg method."""
        p = pburg(segment, order=self.psd_order, scale_by_freq=False)
        psd_vals = p.psd
        freqs = np.array(p.frequencies()) * self.fs
        return psd_vals, freqs

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

    def apply_power_threshold(self, threshold_divisor=10):
        """
        Applies a power threshold to refine the tremor labels.
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
            max_peak_power = axis_data['peak_power'].max()
            threshold = max_peak_power / threshold_divisor
            
            # Apply the threshold to the current axis
            df_features_final.loc[
                (df_features_final['axis'] == axis) & 
                (df_features_final['label'] == 'tremor') & 
                (df_features_final['peak_power'] > threshold),
                'final_label'
            ] = 'tremor'

        # Set the frequency to 1 for all non-tremor windows
        df_features_final.loc[(df_features_final['final_label'] == 'non-tremor'), 'peak_freq_Hz'] = 1

        # Drop the intermediate columns
        df_features_final.drop(columns=['label', 'peak_power'], inplace=True)

        self.df_features_final = df_features_final
        print("Power thresholding and final labeling complete.")
        print("Number of tremor windows detected:", len(self.df_features_final[self.df_features_final['final_label'] == 'tremor']))

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
            if duration_in_seconds > min_duration_s:
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
    specific_file_path = "/Users/patriciawatanabe/Projects/Neurotech/NTUT25_Software/data/accelerometer/raw/801_1_accelerometer.pkl"
    
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