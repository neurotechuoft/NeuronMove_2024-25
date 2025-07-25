#!/Users/marvinlee/anaconda3/envs/pd_tremor/bin/python
import mne
from mne.preprocessing import ICA, compute_current_source_density
from mne_icalabel import label_components
from autoreject import AutoReject

def preprocess_eeg(raw, subject_id=None):
    """All 9 preprocessing steps in one focused function.
    
    Args:
        raw: mne.io.Raw object
        subject_id: Optional ID for logging
        
    Returns:
        Processed epochs (beta-band, tremor channels)
    """
    # 0-1. Montage + CAR
    raw.set_montage('standard_1005').set_eeg_reference('average')
    
    # 2-3. High pass + Notch filter
    raw.filter(0.5, None, method='fir', phase='zero-double')
    raw.notch_filter([60, 120, 180]) #60hz + harmonics
    
    # 4. Epoching
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)
    
    # 5-6. ICA +ICALabel
    ica = ICA(n_components=15, method='infomax', fit_params=dict(extended=True), 
              max_iter=1000, random_state=97)
    ica.fit(epochs)
    ic_labels = label_components(epochs.copy().filter(1, 100), ica, method="iclabel")
    ica.exclude = [idx for idx, prob in enumerate(ic_labels["y_pred_proba"]) if prob < 0.9]
    epochs = ica.apply(epochs)
    
    # 7-9. AutoReject + channel pick + beta filter
    ar = AutoReject(n_interpolate=[1, 4, 32])
    epochs = ar.fit_transform(epochs)

    # 8. Select tremor-relevant channels
    tremor_channels = ['F3', 'F7', 'P4', 'CP2', 'FC6', 'C3', 'C4','Cz']
    epochs.pick_channels(tremor_channels)

    # 9. Isolate Beta Band
    epochs.filter(13, 30, method='fir', fir_design='firwin')
    
    return epochs
