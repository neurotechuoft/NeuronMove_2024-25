import numpy as np
from scipy.signal import firwin, filtfilt, find_peaks, savgol_filter
from spectrum import arburg, arma2psd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from ._01_get_data import DataLoader, MultiDataLoader
from .config import *

class AccelerometerPreprocessor(MultiDataLoader):
    def __init__(self, file_paths, sampling_freq=100):
        # initialize MultiLoader object with accelerometer specific meta data
        super().__init__(
            file_paths,
            meta_data_list=[{
                'Sampling frequency (Hz)': sampling_freq # only takes a constant sampling_freq over all data files
            } for _ in file_paths] # create separate meta data dict for each file
        )
        self.features = []

    
    def dataset_size(self):
        '''
        Prints information about the size of the loaded dataset and a sense of the type of data being dealt with. Pinpoints file 
        paths that have no data loaded, unable to be processed.
        '''
        shapes = [
            data.shape if data is not None else None # Note: .shape doesn't work on everything, it does work on the expected np.array
            for data in self.multi_data[:5]
        ]
        
        valid_data = [data for data in self.multi_data if data is not None] # all not NoneType are considered valid

        none_files = [] # will be populated if there are any NoneType data in self.multi_data
        # gives list of tuples with index (in self.multi_data) and file path for each file associated with NoneType data 
        if len(valid_data) != len(self.multi_data):
            none_files = [
                (i, self.loaders[i].file_path) # Note: self.loaders[i] -> DataLoader object
                for i, val in enumerate(self.multi_data) 
                if val is None
            ] 

        # print the relevant info about the dataset size/type
        print(f'Number of files loaded: {len(self.loaders)}')
        print(f'Data shape for first 5 files: {shapes}') 
        print(f'Valid data (not NoneType): {len(valid_data)} of {len(self.multi_data)}')
        if none_files:
            print(f'NoneType data found in self.multi_data at index (associated file path provided): {none_files}')
    

    def _remove_drift(self, window_size=50):
        '''
        filtfilt is used to removes drift from all loaded data, across all 3 xyz channels. 
        Preserves (3,N) data shape.

        Args:
            window_size: size moving average window (e.g. window_size=50, includes 50 data points)
        '''
        b = np.ones(window_size) / window_size # equal weights over window, assuming data sampled at 100Hz and window_size=50, this means equal weighting over 0.5 seconds
        a = [1.0] # only a[0] <- no feedback from a[1], a[2], ...

        # each data is expected to be (3,N) filter each channel
        for i, data in enumerate(self.multi_data):
            # filtfilt does a forward and backward filter preventing phase distortion and thereby preserving the integrity of the data
            self.multi_data[i] = np.array([filtfilt(b, a, channel) for channel in data]) # cast list of 1D array to 2D array (the original structure)


    def _bandpass_filter(self, lowcut=1.0, highcut=30.0, freq_resolution=1.0):
        '''
        Filters signal to only retain data within the frequency range (Hz), lowcut to highcut.

        Args:
            lowcut: Lower bound of bandpass cutoff (Hz), float type
            highcut: Upper bound of bandpass cutoff (Hz), float type
            freq_resolution: Controls how fine the filter distinguishes between frequencies (in Hz), float type
                (e.g. given 1-30Hz bandpass, 1Hz resolution will gradually attenuate signals until fully attenuating at 0Hz and 31Hz) 
        '''
        
        # fs = [meta_data['Sampling frequency (Hz)'] for meta_data in self.meta_data_list] # get sampling frequency stored in meta data for all loaded files
        
        # get sampling frequency from first data, assume consistent over loaded data
        fs = self.meta_data_list[0]['Sampling frequency (Hz)']
        nyquist_freq = fs/2 # digital filters work relative to nyquist
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        # numtaps ~ sampling freq / resolution, make odd to ensure linear phase i.e. maintaining the timing of data <- otherwise filter can distort the phase timings
        numtaps = round(fs / freq_resolution) if round(fs / freq_resolution) % 2 == 1 else round(fs / freq_resolution) + 1
        # pass_zero=False indicates use of bandpass via firwin, get filter coeffients b
        b = firwin(numtaps, cutoff=[low, high], pass_zero=False)

        # apply filter to each channel for all data <- same process as applying filtfilt in self._remove_drift()
        for i, data in enumerate(self.multi_data):
            self.multi_data[i] = np.array([filtfilt(b, [1.0], channel) for channel in data])
    

    def _segment_data(self, window_size, percent_overlap):
        '''
        Segments data into overlapping windows with Hamming weights.
        Hamming windows taper the edges of the windows to reduce spectral leakage.
        This makes more suitable for Fourier or AR analysis.
        Individual input data shape (3 channels, data points) to individual output data
        shape (3 channels, windows, window size).

        Args:
            window_size: Number of samples per window.
            percent_overlap: Percentage of overlap between consecutive windows.
        '''
        # store to later use to convert windowed data back to timesteps
        self.window_size = window_size
        self.percent_overlap = percent_overlap

        step_size = int(window_size * (1 - percent_overlap/100))
        hamming_window = np.hamming(window_size) # create hamming window of specified size  
                
        for i, data in enumerate(self.multi_data):
            timesteps = data.shape[1]
            n_windows = (timesteps - window_size) // step_size + 1
            
            windowed_data = np.zeros((3, n_windows, window_size)) # hamming to deal with spectral leakage 

            # create all windows
            for j in range(n_windows):
                start_idx = j * step_size
                end_idx = j + step_size

                segment = data[:, start_idx:end_idx] # slice out the jth window across all channels, segment is a 2D array
                windowed_data[:, j, :] = segment * hamming_window[np.newaxis, :] # apply vectorized hamming -> broadcasts hamming 1D array across the 3 channels
            
            self.multi_data[i] = windowed_data # overwrite data with the now windowed data


    def _detect_peak_frequency(self, low_freq=3.0, high_freq=8.0, ar_order=6, prominence_percent=10.0, abs_power_threshold=0.0, 
                               relative_power_threshold_percent=0.0):
        """
        Detects peak frequency using an autoregressive model.
        For each window segment, the AR model is fitted to the data and the one-sided PSD is calculated.
        Peaks found from PSD are thresholded with specified prominence, the power difference between a peak and the lowest 
        point of the surrounding local minima on either side of the peak, relative to the max power in the PSD. Note: Thresholding 
        with prominence helps eliminate low-level vibrations being peaks that don't stand out relative to overall power distribution.
        The peak frequency for each window is the frequency with the highest power (above absolute power threshold) within the 
        frequency range -> low_freq to high_freq (Hz).
        
        Alters self.multi_data -> across all 3 accelerometer channels, all windowed data replaced with the respective dominant 
        frequency, within specified frequency range, or 1 if not found.

        Args:
            low_freq: Lower frequency bound, float type
            high_freq: Upper frequency bound, float type
            ar_order: Order of the autoregressive model (how many past values the current value of a signal depends on), int type
            prominence_percent: Percentage of max power in PSD used as prominence threshold with detecting peaks in PSD, float type
            abs_power_threshold: Peaks will be invalid if its power is below this threshold, generally used if noise floor known, float type
            relative_power_threshold: Peaks will be invalid if power is below this percentage of max power in PSD, float type
        """
        # Note: from _segment_data(), input's individual data shape is (3 channels, windows, window size)
        for i, data in enumerate(self.multi_data):
            peak_data = [[],[],[]] # for efficiency and memory purposes, holding peaks in a list then converting to np array at the end 
            for channel in range(3):
                for window in data[channel]:
                    # get AR coefficients
                    try:
                        ar_coeffs, noise, _ = arburg(window, ar_order)
                    # if error is encountered
                    except Exception as e:
                        raise RuntimeError(
                            f'Error in AR model fitting.\n'
                            f'Window shape: {window.shape}\n'
                            f'Preview of window causing error: {window[:10]}'
                        ) from e
                    
                    # estimate PSD (power spectral density <- frequency domain representation) using ar coefficients
                    fs = self.meta_data_list[0]['Sampling frequency (Hz)'] # sampling frequency
                    sample_interval = 1/fs
                    
                    # arma2psd returns two sided PSD (has both negative of positive frequencies),
                    # accelerometer is a real signal so we only consider positive frequencies (one sided PSD)
                    # since power is symmetric over 0Hz, multiply power by 2 to conserve total power
                    psd_two_sided = arma2psd(ar_coeffs, rho=noise, T=sample_interval)
                    psd_positive_half = psd_two_sided[len(psd_two_sided)//2:].copy()
                    psd_one_sided = psd_positive_half[1:-1] * 2 # double all power except DC (0Hz) and Nyquist (fs/2 Hz)
                    nfft = len(psd_one_sided) 
                    freqs = np.linspace(0, fs/2, nfft) # all frequency bins
                    
                    # get indices of local maxima (peaks) in psd, while excluding peaks with 
                    # prominence less than promience_ratio of max power
                    peaks, _ = find_peaks(psd_one_sided, prominence=psd_one_sided.max() * prominence_percent/100) 

                    if len(peaks) == 0:
                        peak_data[channel].append(1) # 1, given no peaks
                        continue
                    
                    # get frequencies within the specified range, low_freq to high_freq, that have peaks
                    freqs_with_peak = freqs[peaks]
                    freq_idx_mask = (freqs_with_peak > low_freq) & (freqs_with_peak < high_freq) # boolean mask
                    freqs_with_peak = freqs_with_peak[freq_idx_mask]

                    if len(freqs_with_peak) > 0:
                        valid_peaks_idx_mask = np.isin(freqs, freqs_with_peak)
                        # equivalent to: valid_peaks_idx = np.array([True if freq in freqs_with_peak else False for freq in freqs])
                    else:
                        peak_data[channel].append(1) # 1, given no peaks within specified frequency range -> Invalid
                        continue

                    # find frequency associated with valid peak with highest power/PSD amplitude
                    max_power_idx = np.argmax(psd_one_sided[valid_peaks_idx_mask])
                    # apply power threshold (larger of the 2 inputs)
                    power_threshold = max(abs_power_threshold, relative_power_threshold_percent/100 * psd_one_sided.max())

                    if psd_one_sided[max_power_idx] >= power_threshold:
                        peak_data[channel].append(freqs[max_power_idx]) # append peak frequency
                    else:
                        peak_data[channel].append(1) # 1, given peak too low power -> Invalid 
            
            # overwrite the data (for one file)
            self.multi_data[i] = np.array(peak_data)


    def _map_windows_to_timesteps(self):
        """
        Maps window-based dominant frequencies to the original time scale of n_timesteps.
        The dominant frequency for each window segments is upsampled to match the
        length of the original accelerometer data. This results in a time-frequency representation
        of the accelerometer data across the three axes.

        NOTE: Data must have been segmented using _segment_data() prior to calling this method.
        """
        # exits if attributes window_size and percent_overlap do not exist <- assigned in _segment_data()
        if not hasattr(self, 'window_size') or not hasattr(self, 'percent_overlap'):
            print('Attributes window_size and/or percent_overlap not assigned. _segment_data() has not been used.')
            return None
        
        step_size = int(self.window_size * (1 - self.percent_overlap/100)) # compute step size that was initially used in _segment_data()

        for i, data in enumerate(self.multi_data):
            mapped_freq = [None] * 3

            for channel in range(3):

                # create window with all elements being the window's respective dominant frequency
                # e.g. [3,4,...] -> [[3] * window_size, [4] * window_size, ...]
                mapped_freq_1D = [[freq] * self.window_size for freq in data[channel]]
                # flatten the nested list
                mapped_freq_1D = [val for window in channel for val in window]
                mapped_freq[channel] = mapped_freq_1D

            # overwrite the data (for one file)
            self.multi_data[i] = np.array(mapped_freq)
    

    def _smooth_signal(self, smoothing_window):
        '''
        Applies a Savitzky-Golay filter to reduce noise and provide a cleaner output signal for interpretation.

        Args:
            smoothing_window (int): Window used by savgol_filter()
        '''
        self.multi_data = [
            savgol_filter(data, window_length=smoothing_window, polyorder=3, axis=1) # polyorder=2 or 3 for smoothing
            for data in self.multi_data
        ]
    
    def _multiply(self):
        self.multi_data = [np.prod(data, axis=0) for data in self.multi_data]
    
    def _thresholding(self, threshold=3.5):
        self.multi_data = [np.where(data > threshold, 1, 0) for data in self.multi_data]

    def _feature_extraction(self, threshold=3.0):
        """
        Extracts features from the data based on a threshold.

        This method scans through the data to identify segments where the data equals 1.
        It records the start and end indices of these segments and calculates their duration.
        If the duration of a segment exceeds the specified threshold (converted from time steps to seconds),
        the segment is added to the features list.

        Updates self.features with extracted features across all files. Extracted features
        are np.ndarray type with tuples containing the start index, end index, and duration of
        a feature.

        Args:
            threshold (float): The minimum duration (in seconds) for a segment to be considered a feature. Defaults to 3.0
        """
        # convert time threshold to timesteps using sampling frequency
        timestep_threshold = threshold * self.meta_data_list[0]['Sampling frequency (Hz)']

        for data in self.multi_data:
            start = None
            feature_list = []
            for idx, val in enumerate(data):
                if val == 1:
                    if start is None:
                        start = idx
                    # edge case at the end of sequence
                    elif idx == len(data) - 1:
                        if idx - start > timestep_threshold:
                            feature_list.append((start, end, end - start))    
                else:
                    if start is not None:
                        end = idx
                        # apply threshold after converting it to timesteps by multiplying with fs
                        if end - start > timestep_threshold:
                            feature_list.append((start, end, end - start))
                        start = None
            self.features.append(np.array(feature_list))
        
        # if all features have been extracted
        check = 'Feature extraction complete.' if len(self.features) == len(self.multi_data) else 'Feature extraction could not be completed.'
        print(check)


    # --- PLOT and VISUALIZATION methods ---
    # for validation and debugging

    def visualize_features(self, file_idx=0):
        '''
        Plots extracted features from one file specified by file_idx.
        
        Args:
            file_idx (int): Index of file whose features you want to visualize, if not
                specified, will view file at index 0.
        '''
        fs = self.meta_data_list[file_idx]['Sampling frequency (Hz)']
        timesteps = np.arange(self.multi_data[file_idx].shape[1]) / fs

        fig, ax = plt.subplots(1,1)
        line, = ax.plot(timesteps, self.multi_data[file_idx], label='Processed Data')

        for (start, end, duration) in self.features[file_idx]:
            ax.axvspan(start, end, color='r', alpha=0.3) # alpha is transparency
            # mark each feature segment with duration (in seconds)
            ax.text((start + end) / 2, 0.9, f'{(duration / fs): .2f}s', ha='center', fontsize=6, color=(1,0,0,0.5))

        # assign labels
        ax.set_title('Accelerometer Data with Extracted Features')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Feature Presence (1/0)')
        ax.grid(True, alpha=0.5)
        ax.legend()

        plt.show()


    def plot_signal(self, t_start=0.0, t_end=None, indep_var='not labeled', file_idx=0):
        '''
        Plots time-series data in each accelerometer channel independently.
        Will plot a certain time segment if specified, otherwise will plot entire series.
        Plots data from all files, use 'Next' button to go through files sequentially and
        'Exit' to exit the viewing.
        Can plot a single file if its index in self.multi_data is provided.

        Args:
            t_start (float): Start of time segment to plot.
            t_end (float): End of time segment to plot.
            indep_var (str): Name of independent variable across input data.
            file_idx (int): Index of file whose data you want to start the data viewing, if not
                specified, will start at index 0.
        '''
        # get timesteps 
        fs = self.meta_data_list[0]['Sampling frequency (Hz)'] # assuming consistent over all data
        timesteps = np.arange(self.multi_data[0].shape[1]) / fs
        # if end not specified, t_end will default to the end of the series
        t_end = timesteps[-1] if t_end is None else t_end

        # fetch time segment vector
        start_idx = int(t_start * fs)
        end_idx = int(t_end * fs)
        time_seg = timesteps[start_idx:end_idx]

        current_idx = [file_idx] # tracks which plot we are on <- mutuable counter in order to update in callback
        n_files = len(self.multi_data)


        # plot set up
        fig, ax = plt.subplots(3,1, sharex=True)
        plt.subplots_adjust(top=0.9, bottom=0.1) # add space at bottom for buttons
        suptitle = fig.suptitle(f'Accelerometer data -> origin: {self.loaders[current_idx[0]].file_path} index: {current_idx[0]}')

        # initialize plot with first file's data or file specified by file_idx
        data_seg = self.multi_data[current_idx[0]][:, start_idx:end_idx]
        # store each line in independent Line2D object created by .plot() to later overwrite when sequentially plotting each file's data
        line_x, = ax[0].plot(time_seg, data_seg[0])
        line_y, = ax[1].plot(time_seg, data_seg[1])
        line_z, = ax[2].plot(time_seg, data_seg[2])
        # set subplot titles and labels
        ax[0].set_title('X channel')
        ax[1].set_title('Y channel')
        ax[2].set_title('Z channel')
        ax[2].set_xlabel('Time (s)') # same label, implied since sharex=True
        for a in ax:
            a.set_ylabel(indep_var)

        # button setups: Next, Previous, Exit
        ax_next = plt.axes([0.8, 0.025, 0.08, 0.05]) # (x,y) location on plot and (width,height) of button respectively < constructed size starts from bottom-left corner
        btn_next = Button(ax_next, 'Next') # button at position labelled 'Next'

        ax_prev = plt.axes([0.7, 0.025, 0.08, 0.05])
        btn_prev = Button(ax_prev, 'Previous')

        ax_exit = plt.axes([0.9, 0.025, 0.08, 0.05])
        btn_exit = Button(ax_exit, 'Exit')
        
        # nested helper function
        def overwrite_plot(index):
            '''
            Overwrites previous plot with a new file's data/information.
            '''
            # fetch data segment corresponding to time segment
            data_seg = self.multi_data[index][:, start_idx:end_idx]
            # update Line2D objects to overwrite previous plot
            line_x.set_data(time_seg, data_seg[0])
            line_y.set_data(time_seg, data_seg[1])
            line_z.set_data(time_seg, data_seg[2])
            # recalculate axis limits and scale view to bounds for the new data
            for a in ax:
                a.relim()
                a.autoscale_view()
            # overwrite plot title with new file position information
            suptitle.set_text(f'Accelerometer data -> origin: {self.loaders[current_idx[0]].file_path} index: {current_idx[0]}')
            # redraw the plot with all the altered plot parameters (data, etc.)
            fig.canvas.draw_idle()

        # callback functions (from buttons)
        def on_next(event):
            next_idx = current_idx[0] + 1
            if next_idx < n_files:
                # update current_idx and proceed to plot data at the next index
                current_idx[0] = next_idx
                # use helper function to overwrite
                overwrite_plot(current_idx[0])
            else:
                print('End of data preview.')

        def on_prev(event):
            prev_idx = current_idx[0] - 1
            if prev_idx >=0:
                # update current_idx
                current_idx[0] = prev_idx
                # overwrite
                overwrite_plot(current_idx[0])
            else:
                print('Start of data preview.')

        def on_exit(event):
            plt.close(fig)

        # bind callback functions to the buttons
        btn_next.on_clicked(on_next)
        btn_exit.on_clicked(on_exit)
        btn_prev.on_clicked(on_prev)

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    try:
        file_paths = list(DATA_DIR.glob("*.pkl"))
        if not file_paths:
            raise FileNotFoundError("No .pkl files found in the specified directory.")
    except FileNotFoundError as e:
        print(e)
        print("Please ensure the data directory contains accelerometer data files.")
        file_paths = []
    
    if file_paths:
        preprocessor = AccelerometerPreprocessor(file_paths, sampling_freq=SAMPLING_FREQ)
        print("AccelerometerPreprocessor instantiatead.")
        preprocessor.dataset_size()

        print("Removing signal drift...")
        preprocessor._remove_drift(window_size=DRIFT_WINDOW_SIZE)

        print("Applying bandpass filter...")
        preprocessor._bandpass_filter(lowcut=BANDPASS_LOWCUT, highcut=BANDPASS_HIGHCUT, freq_resolution=BANDPASS_FREQ_RESOLUTION)

        print("Segmenting data...")
        preprocessor._segment_data(window_size=SEGMENT_WINDOW_SIZE, percent_overlap=SEGMENT_PERCENT_OVERLAP)

        print("Detecting peak frequencies in windows...")
        preprocessor._detect_peak_frequency(
            low_freq=PEAK_LOW_FREQ, 
            high_freq=PEAK_HIGH_FREQ, 
            ar_order=AR_ORDER, 
            prominence_percent=PROMINENCE_PERCENT, 
            abs_power_threshold=ABS_POWER_THRESHOLD, 
            relative_power_threshold_percent=RELATIVE_POWER_THRESHOLD_PERCENT
        )

        print("Mapping frequencies back to time series...")
        preprocessor._map_windows_to_timesteps()

        print("Smoothing signal...")
        preprocessor._smooth_signal(smoothing_window=SMOOTHING_WINDOW)

        print("Multiplying across channels...")
        preprocessor._multiply()

        print("Applying thresholding...")
        preprocessor._thresholding(threshold=FEATURE_THRESHOLD)

        print("Extracting features...")
        preprocessor._feature_extraction(threshold=FEATURE_EXTRACTION_DURATION_THRESHOLD)

        print("Visualizing features...")
        preprocessor.visualize_features(file_idx=0)
    else:
        print("No files to process. Please check the data directory.")
