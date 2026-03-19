import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt

FS_TARGET = 100  # Hz (the dataset has no timestamps, so 100Hz was assumed as standard sampling)

def drift_remove_ma(x: np.ndarray, fs: int = FS_TARGET, seconds: float = 5.0) -> np.ndarray:

    L = int(fs * seconds)  
    L = max(L, 3)
    ma = np.ones(L, dtype=float) / L
    trend = filtfilt(ma, [1.0], x)  
    return x - trend

def bandpass_fir(x: np.ndarray, fs: int = FS_TARGET, lo: float = 1.0, hi: float = 30.0, numtaps: int = 201) -> np.ndarray:

    bp = firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)
    return filtfilt(bp, [1.0], x)

def window_xyz(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    y: np.ndarray | None = None,
    fs: int = FS_TARGET,
    win_sec: float = 3.0,
    overlap: float = 0.9,
):

    win = int(fs * win_sec)                  # 300
    hop = int(win * (1.0 - overlap))         # 30
    hop = max(hop, 1)

    ham = np.hamming(win).astype(np.float32)

    X = []
    y_out = [] if y is not None else None

    for i in range(0, len(ax) - win + 1, hop):
        seg = np.stack([ax[i:i+win], ay[i:i+win], az[i:i+win]], axis=1)  # (win,3)
        seg = seg * ham[:, None]
        X.append(seg.astype(np.float32))

        if y is not None:
            lab = y[i:i+win]
            y_out.append(int(np.bincount(lab).argmax()))

    X = np.asarray(X, dtype=np.float32)
    if y is None:
        return X
    return X, np.asarray(y_out, dtype=np.int64)

def preprocess_and_window(
    csv_path: str,
    accel_cols=("aX", "aY", "aZ"),
    label_col="Result",
    fs: int = FS_TARGET,
    win_sec: float = 3.0,
    overlap: float = 0.9,
):

    df = pd.read_csv(csv_path)

    ax = df[accel_cols[0]].to_numpy(dtype=float)
    ay = df[accel_cols[1]].to_numpy(dtype=float)
    az = df[accel_cols[2]].to_numpy(dtype=float)
    y  = df[label_col].to_numpy(dtype=int)

    ax = drift_remove_ma(ax, fs=fs, seconds=5.0)
    ay = drift_remove_ma(ay, fs=fs, seconds=5.0)
    az = drift_remove_ma(az, fs=fs, seconds=5.0)

    ax = bandpass_fir(ax, fs=fs, lo=1.0, hi=30.0, numtaps=201)
    ay = bandpass_fir(ay, fs=fs, lo=1.0, hi=30.0, numtaps=201)
    az = bandpass_fir(az, fs=fs, lo=1.0, hi=30.0, numtaps=201)

    X, y_win = window_xyz(ax, ay, az, y=y, fs=fs, win_sec=win_sec, overlap=overlap)
    return X, y_win