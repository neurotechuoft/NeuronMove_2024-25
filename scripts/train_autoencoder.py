# ── How to run ────────────────────────────────────────────────────────────────
#   python scripts/train_autoencoder.py --dataset kaggle
#   python scripts/train_autoencoder.py --dataset alameda
#   python scripts/train_autoencoder.py --dataset pads
#
# PADS (preprocessed movement *.ml.bin):
#   python scripts/train_autoencoder.py --dataset pads --epochs 50
#   Requires: preprocessed/stratified_subset_file_list.csv (or file_list.csv) + preprocessed/movement/{id:03d}_ml.bin
#   Splits train/val/test by patient id BEFORE loading bins. Saves weights + norm stats to models/autoencoder_pads.pt
#
# Notes:
#   - Kaggle: random row split (no subject id in CSV)
#   - ALAMEDA: split by subject_id (70/15/15)
#   - PADS: split by id with GroupShuffleSplit (~60/20/20); Conv1D AE on (time × channels)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parents[1]
PREP_DIR = BASE_DIR / "preprocessed"
MOV_DIR = PREP_DIR / "movement"
N_TIME = 976
LOGS_DIR = BASE_DIR / "logs"
AE_RESULTS_JSON = LOGS_DIR / "ae_results.json"
AE_LAST_RUN_JSON = LOGS_DIR / "ae_last_run.json"


def save_ae_results_json(run: dict) -> None:
    """Append one run to logs/ae_results.json and overwrite logs/ae_last_run.json."""
    run = {**run, "saved_at_utc": datetime.now(timezone.utc).isoformat()}
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    history: list = []
    if AE_RESULTS_JSON.exists():
        try:
            with AE_RESULTS_JSON.open(encoding="utf-8") as f:
                loaded = json.load(f)
            history = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(run)
    with AE_RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with AE_LAST_RUN_JSON.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    print(f"Results JSON: {AE_RESULTS_JSON} (appended), {AE_LAST_RUN_JSON} (latest run)")

DATASETS = {
    "kaggle": {
        "path": "data/external/kaggle_mpu9250/Dataset.csv",
        "label_col": "Result",
        "drop_cols": [],
        "meta_cols": [],
        "split_by": "random",
        "bottleneck": 8,
        "save_as": "models/autoencoder_kaggle.pt",
    },
    "alameda": {
        "path": "data/external/zenodo_alameda/ALAMEDA_PD_tremor_dataset.csv",
        "label_col": [
            "Constancy_of_rest",
            "Kinetic_tremor",
            "Postural_tremor",
            "Rest_tremor",
        ],
        "drop_cols": [],
        "meta_cols": ["subject_id", "start_timestamp", "end_timestamp"],
        "split_by": "subject_id",
        "bottleneck": 16,
        "save_as": "models/autoencoder_alameda.pt",
    },
    "pads": {
        "path": None,
        "bottleneck": 32,
        "save_as": "models/autoencoder_pads.pt",
    },
    "new_mexico": {
        "path": "data/external/new_mexico/",
        "label_col": None,
        "drop_cols": [],
        "meta_cols": [],
        "split_by": "random",
        "bottleneck": 16,
        "save_as": "models/autoencoder_new_mexico.pt",
    },
}


def split_train_val_test_by_patient(
    patient_ids: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices; no patient appears in more than one split."""
    n = len(patient_ids)
    idx = np.arange(n)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(idx, groups=patient_ids))
    tv = patient_ids[train_val_idx]
    frac_val_of_tv = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=frac_val_of_tv, random_state=random_state + 1
    )
    rel_train, rel_val = next(
        gss2.split(np.zeros(len(train_val_idx)), groups=tv)
    )
    train_idx = train_val_idx[rel_train]
    val_idx = train_val_idx[rel_val]
    return train_idx, val_idx, test_idx


def resolve_pads_csv() -> Path:
    strat = PREP_DIR / "stratified_subset_file_list.csv"
    full = PREP_DIR / "file_list.csv"
    if strat.exists():
        return strat
    if full.exists():
        return full
    raise FileNotFoundError(
        f"PADS needs {strat} or {full} (from PADS preprocessing)."
    )


def load_bins_for_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load *_ml.bin rows as (N, T, C). Skips missing files."""
    xs: list[np.ndarray] = []
    pids: list[int] = []
    for _, row in df.iterrows():
        pid = int(row["id"])
        bin_path = MOV_DIR / f"{pid:03d}_ml.bin"
        if not bin_path.exists():
            continue
        raw = np.fromfile(bin_path, dtype=np.float32)
        if raw.size % N_TIME != 0:
            continue
        n_ch = raw.size // N_TIME
        mat = raw.reshape(n_ch, N_TIME).T.astype(np.float32)
        xs.append(mat)
        pids.append(pid)
    if not xs:
        raise RuntimeError("No valid PADS *.ml.bin files found for this dataframe.")
    c0 = xs[0].shape[1]
    pairs = [(a, p) for a, p in zip(xs, pids) if a.shape[1] == c0]
    if not pairs:
        raise RuntimeError("Inconsistent channel counts across PADS bins.")
    xs = [a for a, _ in pairs]
    pids = [p for _, p in pairs]
    X = np.stack(xs, axis=0)
    return X, np.array(pids, dtype=np.int64)


def load_pads_train_val_test(
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    csv_path = resolve_pads_csv()
    print(f"PADS file list: {csv_path}")
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        raise ValueError("PADS CSV must have an 'id' column.")
    patient_ids = df["id"].values.astype(np.int64)
    train_idx, val_idx, test_idx = split_train_val_test_by_patient(
        patient_ids, test_size=test_size, val_size=val_size, random_state=seed
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(
        f"PADS split by patient — list rows: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}"
    )
    X_train, _ = load_bins_for_df(train_df)
    X_val, _ = load_bins_for_df(val_df)
    X_test, _ = load_bins_for_df(test_df)
    n_ch = X_train.shape[2]
    print(
        f"Loaded windows — train {X_train.shape[0]}, val {X_val.shape[0]}, test {X_test.shape[0]}; shape (N,T,C)=(_, {N_TIME}, {n_ch})"
    )
    return X_train, X_val, X_test, n_ch


class Autoencoder(nn.Module):
    """Dense AE for flat feature vectors (Kaggle / ALAMEDA)."""

    def __init__(self, input_dim: int, bottleneck: int):
        super().__init__()
        h1 = max(bottleneck * 4, 32)
        h2 = max(bottleneck * 2, 16)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PadsConv1dAE(nn.Module):
    """1D conv autoencoder on (batch, time, channels)."""

    def __init__(self, n_channels: int, n_time: int, bottleneck: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_time = n_time
        self.enc_conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 8, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        t_down = n_time // 8
        self.flat_dim = t_down * 8
        self.fc_enc = nn.Linear(self.flat_dim, bottleneck)
        self.fc_dec = nn.Linear(bottleneck, self.flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        z = x.transpose(1, 2)
        z = self.enc_conv(z)
        b = z.size(0)
        zf = z.reshape(b, -1)
        latent = self.fc_enc(zf)
        zd = self.fc_dec(latent).view(b, 8, -1)
        out = self.dec_conv(zd)
        out = out.transpose(1, 2)
        return out


def load_data(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = BASE_DIR / cfg["path"]
    df = pd.read_csv(path)
    mag_cols = [
        c
        for c in df.columns
        if any(m in c.lower() for m in ["magx", "magy", "magz"])
    ]
    drop = mag_cols + cfg["drop_cols"]
    label_col = cfg["label_col"]
    meta_cols = cfg["meta_cols"]
    if isinstance(label_col, list):
        exclude = drop + label_col + meta_cols
    elif label_col:
        exclude = drop + [label_col] + meta_cols
    else:
        exclude = drop + meta_cols
    X = df.drop(columns=exclude, errors="ignore").values.astype(np.float32)
    if isinstance(label_col, list):
        y = (df[label_col].max(axis=1) > 0).astype(int).values
    elif label_col:
        y = df[label_col].values
    else:
        y = np.zeros(len(X))
    return df, X, y


def split_data(df: pd.DataFrame, X: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(42)
    if cfg["split_by"] == "subject_id":
        subjects = df["subject_id"].unique()
        np.random.shuffle(subjects)
        n = len(subjects)
        train_s = subjects[: int(n * 0.70)]
        val_s = subjects[int(n * 0.70) : int(n * 0.85)]
        test_s = subjects[int(n * 0.85) :]
        train_mask = df["subject_id"].isin(train_s).values
        val_mask = df["subject_id"].isin(val_s).values
        test_mask = df["subject_id"].isin(test_s).values
        print(f"Subjects — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")
    else:
        idx = np.random.permutation(len(X))
        t, v = int(len(X) * 0.70), int(len(X) * 0.85)
        train_mask = np.zeros(len(X), dtype=bool)
        val_mask = np.zeros(len(X), dtype=bool)
        test_mask = np.zeros(len(X), dtype=bool)
        train_mask[idx[:t]] = True
        val_mask[idx[t:v]] = True
        test_mask[idx[v:]] = True
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    print(
        f"Windows  — train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )
    return X_train, X_val, X_test


def train_dense(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    bottleneck: int,
    save_as: Path,
    epochs: int = 30,
    dataset_name: str = "unknown",
    split_info: dict | None = None,
) -> None:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=64,
    )
    model = Autoencoder(input_dim, bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch_log: list[dict] = []
    final_train_mse = 0.0
    final_val_mse = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss += loss_fn(model(batch), batch).item()
        tr = train_loss / len(train_loader)
        va = val_loss / len(val_loader)
        final_train_mse = tr
        final_val_mse = va
        epoch_log.append({"epoch": epoch + 1, "train_mse": tr, "val_mse": va})
        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {tr:.4f} | Val Loss: {va:.4f}"
        )
    (BASE_DIR / save_as).parent.mkdir(parents=True, exist_ok=True)
    model_path = str(BASE_DIR / save_as)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    save_ae_results_json(
        {
            "dataset": dataset_name,
            "model": "dense_linear_ae",
            "epochs": epochs,
            "input_dim": input_dim,
            "bottleneck": bottleneck,
            "device": str(device),
            "torch": torch.__version__,
            "n_train_windows": int(X_train.shape[0]),
            "n_val_windows": int(X_val.shape[0]),
            "split": split_info or {},
            "final_train_mse": float(final_train_mse),
            "final_val_mse": float(final_val_mse),
            "epoch_history": epoch_log,
            "checkpoint_path": model_path,
        }
    )


def train_pads(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_channels: int,
    bottleneck: int,
    save_as: Path,
    epochs: int = 30,
    batch_size: int = 16,
) -> None:
    """Per-channel norm fit on train only; MSE in normalized space."""
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sig = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_n = (X_train - mu) / sig
    X_val_n = (X_val - mu) / sig
    X_test_n = (X_test - mu) / sig

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_n, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_n, dtype=torch.float32)),
        batch_size=batch_size,
    )

    n_time = X_train.shape[1]
    model = PadsConv1dAE(n_channels, n_time, bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epoch_log: list[dict] = []
    final_train_mse = 0.0
    final_val_mse = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss += loss_fn(model(batch), batch).item()
        tr = train_loss / len(train_loader)
        va = val_loss / len(val_loader)
        final_train_mse = tr
        final_val_mse = va
        epoch_log.append({"epoch": epoch + 1, "train_mse": tr, "val_mse": va})
        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {tr:.4f} | Val Loss: {va:.4f}"
        )

    model.eval()
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_n, dtype=torch.float32)),
        batch_size=batch_size,
    )
    test_loss = 0.0
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            test_loss += loss_fn(model(batch), batch).item()
    test_mse = test_loss / len(test_loader)
    print(f"Test MSE (normalized): {test_mse:.6f}")

    out = BASE_DIR / save_as
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mu": mu,
            "sig": sig,
            "n_channels": n_channels,
            "n_time": n_time,
            "bottleneck": bottleneck,
        },
        out,
    )
    print(f"Saved PADS autoencoder + norm stats to {out}")

    save_ae_results_json(
        {
            "dataset": "pads",
            "model": "conv1d_ae",
            "epochs": epochs,
            "batch_size": batch_size,
            "n_time": int(n_time),
            "n_channels": int(n_channels),
            "bottleneck": bottleneck,
            "device": str(device),
            "torch": torch.__version__,
            "n_train_windows": int(X_train.shape[0]),
            "n_val_windows": int(X_val.shape[0]),
            "n_test_windows": int(X_test.shape[0]),
            "split": {"by": "patient_id", "method": "GroupShuffleSplit ~60/20/20"},
            "final_train_mse": float(final_train_mse),
            "final_val_mse": float(final_val_mse),
            "test_mse_normalized": float(test_mse),
            "epoch_history": epoch_log,
            "checkpoint_path": str(out),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASETS.keys()), help="dataset name"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16, help="PADS only")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]

    if args.dataset == "pads":
        print("\nLoading PADS preprocessed *.ml.bin (patient split before load)...")
        X_train, X_val, X_test, n_ch = load_pads_train_val_test()
        print(f"Bottleneck: {cfg['bottleneck']}")
        train_pads(
            X_train,
            X_val,
            X_test,
            n_ch,
            cfg["bottleneck"],
            Path(cfg["save_as"]),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.dataset == "new_mexico":
        raise NotImplementedError(
            "new_mexico: add path + loader when mk/Daniel provide formatted xyz."
        )
    else:
        path = BASE_DIR / cfg["path"]
        print(f"\nLoading {args.dataset} from {path}")
        df, X, _y = load_data(cfg)
        X_train, X_val, _ = split_data(df, X, cfg)
        input_dim = X_train.shape[1]
        print(f"Input dim: {input_dim}, Bottleneck: {cfg['bottleneck']}")
        split_info = {
            "method": cfg.get("split_by", "random"),
            "description": "70/15/15 by subject_id" if cfg.get("split_by") == "subject_id" else "70/15/15 random rows",
        }
        train_dense(
            X_train,
            X_val,
            input_dim,
            cfg["bottleneck"],
            Path(cfg["save_as"]),
            epochs=args.epochs,
            dataset_name=args.dataset,
            split_info=split_info,
        )
