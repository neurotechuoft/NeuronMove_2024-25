# NTUT25_SOFTWARE/src/eeg/eeg_preprocessor.py

import mne
import numpy as np
import os
import scipy.stats
from mne.annotations import Annotations

class EEGProcessor:
    """
    A class to encapsulate the preprocessing steps for a single EEG subject/session,
    based on the SOP document.
    """

    def __init__(self, subj_id, session_num, config):
        """
        Initializes the EEGProcessor for a specific subject and session.

        Parameters:
        ----------
        subj_id : int
            The subject ID.
        session_num : int
            The session number (1 or 2).
        config : module
            The config module containing all global parameters.
        """
        self.subj_id = subj_id
        self.session_num = session_num
        self.config = config

        self.raw = None  # To hold the raw MNE object (all channels)
        self.raw_original_all_channels = None # A copy before any dropping/picking for key steps
        self.epochs = None # To hold the epoched MNE object
        self.epochs_eeg_only = None # To hold the final EEG-only epoched object

        # Construct file paths
        self.task_name = "REST" # Consistent with your current data filenames
        self.raw_fname_base = f"{self.subj_id}_{self.session_num}_PD_{self.task_name}"
        self.raw_fname_full = os.path.join(self.config.RAW_EEG_DATA_DIR, f"{self.raw_fname_base}{self.config.RAW_FNAME_SUFFIX}")
        self.processed_fname = os.path.join(self.config.PROCESSED_EEG_DATA_DIR, f"{self.raw_fname_base}_processed{self.config.RAW_FNAME_SUFFIX}")

        print(f"\n--- Initializing Processor for Subject: {self.subj_id}, Session: {self.session_num} ---")

    def _load_and_merge_raw_data(self):
        """
        Loads the raw .fif data for the subject/session, handling the merge case for subj_id 810.
        (Consolidates SOP 4.1).
        """
        print(f"Loading raw file: {os.path.basename(self.raw_fname_full)}")

        if self.subj_id == 810 and self.session_num == 1:
            raw_fname_b_base = f"{self.subj_id}_{self.session_num}_PD_{self.task_name}b" 
            raw_fname_b_full = os.path.join(self.config.RAW_EEG_DATA_DIR, f"{raw_fname_b_base}{self.config.RAW_FNAME_SUFFIX}")
            try:
                raw_a = mne.io.read_raw_fif(self.raw_fname_full, preload=True, verbose=False)
                raw_b = mne.io.read_raw_fif(raw_fname_b_full, preload=True, verbose=False)
                self.raw = mne.concatenate_raws([raw_a, raw_b])
                print(f"Merged {os.path.basename(self.raw_fname_full)} and {os.path.basename(raw_fname_b_full)}")
            except FileNotFoundError:
                raise FileNotFoundError(f"One or both files for {self.subj_id}_{self.session_num} merge not found.")
        else:
            try:
                self.raw = mne.io.read_raw_fif(self.raw_fname_full, preload=True, verbose=False)
                print(f"Loaded {os.path.basename(self.raw_fname_full)}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Raw file {self.raw_fname_full} not found.")
            except Exception as e:
                raise Exception(f"Error loading {self.raw_fname_full}: {e}")

        # This copy will be used for operations that need all channels
        self.raw_original_all_channels = self.raw.copy() 

    def _apply_filtering_and_rerferencing(self):
        """
        Applies filtering, line noise removal, montage, and re-referencing per SOP.
        (Consolidates SOP 4.5, 4.6, 4.7, 4.9).
        """
        print("\n--- Applying Filtering, Line Noise Removal, and Re-referencing ---")

        # SOP 4.6 & 4.5: Bandpass Filtering (FIR 0.5-50 Hz)
        # This single call removes slow drifts (4.5) and high-frequency noise/line noise (4.6)
        if self.raw_original_all_channels.get_channel_types(picks='eeg'):
            print(f"Applying FIR bandpass filter: {self.config.HIGH_PASS_FREQ}-{self.config.LOW_PASS_FREQ} Hz.")
            self.raw_original_all_channels.filter(l_freq=self.config.HIGH_PASS_FREQ, h_freq=self.config.LOW_PASS_FREQ, 
                                          fir_window='hamming', verbose=False)
        
        # SOP 4.7: Line Noise Removal (Notch filter)
        if self.config.LINE_FREQ is not None and self.config.LINE_FREQ > 0:
            print(f"Applying notch filter at {self.config.LINE_FREQ} Hz.")
            self.raw_original_all_channels.notch_filter(self.config.LINE_FREQ, verbose=False)

        # SOP 4.9: Re-referencing
        if self.raw_original_all_channels.get_channel_types(picks='eeg'):
            print(f"Applying average reference to EEG channels.")
            self.raw_original_all_channels.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
            self.raw_original_all_channels.apply_proj() 
        else:
            print("No EEG channels found for re-referencing. Skipping average reference.")

        # SOP 4.2 (part of it): Set standard montage
        try:
            montage = mne.channels.make_standard_montage('standard_1005') 
            self.raw_original_all_channels.set_montage(montage, on_missing='ignore') 
            print(f"Set montage.")
        except Exception as e:
            print(f"Warning: Could not set standard montage on raw_full_channels: {e}. Topoplots may not work.")

    def _segment_data(self):
        """
        Segments continuous REST data into 3-second epochs based on SOP 4.11.
        (Adapts from tremor labels to eyes-open/closed triggers).
        """
        print("\n--- Segmenting Continuous Data into 3-Second Epochs ---")
        
        # SOP 4.11.1: Define Events for Epoching
        events, event_id_from_raw = mne.events_from_annotations(self.raw_original_all_channels, event_id=self.config.EVENT_ID)

        # We will create fixed-duration epochs based on the start of the eyes-open/closed segments
        eyes_open_event_ids = [self.config.EVENT_ID['1'], self.config.EVENT_ID['2']]
        eyes_closed_event_ids = [self.config.EVENT_ID['3'], self.config.EVENT_ID['4']]
        
        eyes_open_events = events[np.isin(events[:, 2], eyes_open_event_ids)]
        eyes_closed_events = events[np.isin(events[:, 2], eyes_closed_event_ids)]
        
        # Create a new set of events representing 3-second, non-overlapping segments
        epoch_duration_samples = int(self.raw_original_all_channels.info['sfreq'] * 3.0)
        
        # Find start of eyes-open and eyes-closed blocks
        eyes_open_start_sample = eyes_open_events[0, 0] if len(eyes_open_events) > 0 else 0
        eyes_closed_start_sample = eyes_closed_events[0, 0] if len(eyes_closed_events) > 0 else 0
        
        new_events = []
        # SOP 4.11.2: Create Epochs
        for start_sample in np.arange(eyes_closed_start_sample, eyes_closed_start_sample + 60*self.raw_original_all_channels.info['sfreq'], epoch_duration_samples):
             if start_sample < len(self.raw_original_all_channels.times) - epoch_duration_samples:
                new_events.append([int(start_sample), 0, 200]) # Event ID 200 for 'Eyes_Closed'
        
        for start_sample in np.arange(eyes_open_start_sample, eyes_open_start_sample + 60*self.raw_original_all_channels.info['sfreq'], epoch_duration_samples):
            if start_sample < len(self.raw_original_all_channels.times) - epoch_duration_samples:
                new_events.append([int(start_sample), 0, 100]) # Event ID 100 for 'Eyes_Open'
        
        new_events = np.array(new_events)
        if len(new_events) > 0:
            # Sort events by time
            new_events = new_events[new_events[:, 0].argsort()]
            new_event_id = {'Eyes_Open': 100, 'Eyes_Closed': 200}
            
            self.epochs = mne.Epochs(self.raw_original_all_channels, new_events, event_id=new_event_id, 
                                        tmin=0, tmax=3.0, baseline=None, preload=True, verbose=False)
            print(f"Segmented into {len(self.epochs['Eyes_Open'])} 'Eyes_Open' epochs and {len(self.epochs['Eyes_Closed'])} 'Eyes_Closed' epochs.")
        else:
            raise ValueError("No valid events found in raw.annotations for segmentation.")


    def _apply_apple_logic(self):
        """
        Replicates the core logic of APPLE_PDDys (bad channel/epoch/ICA detection/correction).
        (Consolidates SOP 4.8, 4.10, 4.12).
        """
        print("\n--- Applying APPLE-like Pre-processing ---")
        
        eeg_ch_names_for_ica = self.raw_original_all_channels.copy().pick_types(eeg=True, exclude='bads').ch_names

        # --- 1. Automated Bad Channel Detection & Interpolation (SOP 4.8) ---
        temp_epochs_for_bad_ch_detection = self.epochs.copy().pick_types(eeg=True, exclude='bads')
        
        bad_channels_detected = []
        if temp_epochs_for_bad_ch_detection.ch_names: 
            channel_stds = np.std(temp_epochs_for_bad_ch_detection.get_data(), axis=(0, 2))
            flat_ch_indices = np.where(channel_stds < self.config.BAD_CH_FLAT_THRESHOLD_UV)[0] 
            if len(flat_ch_indices) > 0:
                flat_ch_names = [temp_epochs_for_bad_ch_detection.ch_names[i] for i in flat_ch_indices]
                print(f"Detected flat channels: {flat_ch_names}")
                bad_channels_detected.extend(flat_ch_names)

            if len(temp_epochs_for_bad_ch_detection.ch_names) > 1 and len(channel_stds) > 1:
                channel_stds_z = np.abs(scipy.stats.zscore(channel_stds))
                noisy_ch_indices_z = np.where(channel_stds_z > self.config.BAD_CH_NOISY_Z_THRESHOLD)[0]
                if len(noisy_ch_indices_z) > 0:
                    noisy_ch_names_z = [temp_epochs_for_bad_ch_detection.ch_names[i] for i in noisy_ch_indices_z]
                    print(f"Detected noisy channels (high std): {noisy_ch_names_z}")
                    bad_channels_detected.extend(noisy_ch_names_z)
            
        self.bad_channels_found = list(np.unique(bad_channels_detected))
        
        epochs_cleaned = self.epochs.copy()
        if self.bad_channels_found:
            print(f"Identified bad channels for interpolation: {self.bad_channels_found}")
            self.raw_original_all_channels.info['bads'].extend(self.bad_channels_found) 
            self.raw_original_all_channels.info['bads'] = list(np.unique(self.raw_original_all_channels.info['bads']))
            
            self.raw_original_all_channels.interpolate_bads(reset_bads=False, verbose=False) 
            self.epochs = epochs_initial.copy().interpolate_bads(reset_bads=True, verbose=False) 
        else:
            print("No bad channels identified for interpolation.")
            
        # --- 2. Artifact Rejection (Segment-level) (SOP 4.12) ---
        epochs_before_rejection = len(self.epochs)
        self.epochs.drop_bad(reject=dict(eeg=self.config.BAD_EPOCH_PEAK_TO_PEAK_UV), 
                            flat=dict(eeg=self.config.BAD_EPOCH_FLAT_UV), 
                            verbose=False)
        
        all_epoch_indices = np.arange(epochs_before_rejection)
        kept_epoch_indices = self.epochs.selection
        self.bad_epochs_indices_found = np.setdiff1d(all_epoch_indices, kept_epoch_indices).tolist()
        
        print(f"Rejected {epochs_before_rejection - len(self.epochs)} epochs based on amplitude thresholds and flatness.")
        
        # --- 3. ICA for Artifact Removal (SOP 4.10) ---
        raw_for_ica = self.raw_original_all_channels.copy().filter(l_freq=self.config.ICA_HIGH_PASS_FREQ, h_freq=None, verbose=False)

        ica = mne.preprocessing.ICA(n_components=self.config.ICA_N_COMPONENTS, method=self.config.ICA_METHOD, 
                                    random_state=self.config.ICA_RANDOM_STATE, max_iter='auto', verbose=False)
        print("Fitting ICA (this may take a moment)...")
        ica.fit(raw_for_ica, picks=eeg_ch_names_for_ica) 

        self.bad_ica_components_found = []
        if self.config.VEOG_CHANNEL_NAME in self.raw_original_all_channels.ch_names: 
            eog_indices, scores = ica.find_bads_eog(self.raw_original_all_channels, ch_name=self.config.VEOG_CHANNEL_NAME, measure='correlation', threshold='auto', verbose=False)
            if eog_indices:
                print(f"Automatically detected EOG components: {eog_indices}")
                self.bad_ica_components_found.extend(eog_indices)
            else:
                print(f"No EOG components automatically detected from '{self.config.VEOG_CHANNEL_NAME}'.")
        else:
            print(f"Warning: VEOG channel '{self.config.VEOG_CHANNEL_NAME}' not found in raw data. Skipping automatic EOG detection via dedicated channel.")
            
        self.bad_ica_components_found = list(np.unique(self.bad_ica_components_found))

        if self.bad_ica_components_found:
            print(f"Excluding ICA components: {self.bad_ica_components_found} from epochs.")
            self.epochs = ica.apply(self.epochs.copy(), exclude=self.bad_ica_components_found, verbose=False)
        else:
            print("No ICA components excluded automatically.")

        print("--- APPLE-like Pre-processing Complete ---")

    def _final_channel_picking_and_save(self):
        # The SOP refers to 64-channels, but your MATLAB data had 67 channels
        # and the MATLAB script's output was 63 channels. Let's stick with that.
        print("\n--- Final Channel Picking and Saving ---")
        self.epochs_eeg_only = self.epochs.copy().pick_types(eeg=True)
        print(f"Final epochs object contains {len(self.epochs_eeg_only.ch_names)} EEG channels.")

        print(f"Saving cleaned epochs to {self.processed_fname}")
        self.epochs_eeg_only.save(self.processed_fname, overwrite=self.config.OVERWRITE_PROCESSED_FILES, verbose=False) 

    def process_subject(self):
        """
        Orchestrates the full preprocessing pipeline for the initialized subject and session.
        """
        if os.path.exists(self.processed_fname) and not self.config.OVERWRITE_PROCESSED_FILES:
            print(f"Skipping {self.subj_id}_Session{self.session_num}: Processed file already exists at {self.processed_fname}")
            return 

        try:
            self._load_and_merge_raw_data()
            self._apply_initial_channel_management()
            
            # --- New SOP-based processing flow ---
            self._apply_filtering_and_rerferencing()
            self._segment_data()
            self._apply_apple_logic()
            self._final_channel_picking_and_save()

            print(f"Finished processing {self.subj_id}_Session{self.session_num}. Saved to {self.processed_fname}")

        except FileNotFoundError as e:
            print(f"Error for {self.subj_id}_Session{self.session_num}: {e}")
        except ValueError as e:
            print(f"Error for {self.subj_id}_Session{self.session_num}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {self.subj_id}_Session{self.session_num}: {e}")
            import traceback
            traceback.print_exc()