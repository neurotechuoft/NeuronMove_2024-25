import numpy as np
from scipy.signal import firwin, filtfilt, find_peaks
from spectrum import pburg
from get_data import DataLoader, MultiDataLoader

class AccelerometerPreprocessor(MultiDataLoader):
    def __init__(self, file_paths, sampling_freq=100):
        # initialize MultiLoader object with accelerometer specific meta data
        super().__init__(
            file_paths,
            meta_data_list=[{
                'Sampling frequency (Hz)': sampling_freq # only takes a constant sampling_freq over all data files
            } for _ in file_paths] # create separate meta data dict for each file
        )
        self.features = None

    
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
            lowcut: lower bound of bandpass cutoff (Hz), float type
            highcut: upper bound of bandpass cutoff (Hz), float type
            freq_resolution: controls how fine the filter distinguishes between frequencies (in Hz), float type
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
    
    def _segment_data(self, window_size, overlap_ratio):
        raise NotImplementedError('This function is not yet implemented. Please look forward to its completion.')
    
    def _detect_peak_frequency(self, low_freq=3, high_freq=8, ar_order=6):
        raise NotImplementedError('This function is not yet implemented. Please look forward to its completion.')
    
    # freq-time representation
    # feature extraction
    # visualization tools (maybe hold it in a class)