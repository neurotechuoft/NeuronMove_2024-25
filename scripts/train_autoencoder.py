# ── How to run ────────────────────────────────────────────────────────────────
# Basic usage:
#   python scripts/train_autoencoder.py --dataset <name>
#
# Available datasets:
#   --dataset kaggle       Kaggle MPU9250 hand tremor (~28k rows, binary label)
#   --dataset alameda      Zenodo ALAMEDA (~4152 windows, 92 features, split by subject)
#   --dataset pads         PADS smartwatch (unlabeled pretrain, not ready yet)
#   --dataset new_mexico   New Mexico pool (unlabeled, waiting on Daniel)
#
# Custom epochs (default is 30):
#   python scripts/train_autoencoder.py --dataset kaggle --epochs 50
#   python scripts/train_autoencoder.py --dataset alameda --epochs 100
#
# Models save to:
#   models/autoencoder_kaggle.pt
#   models/autoencoder_alameda.pt
#   models/autoencoder_pads.pt
#   models/autoencoder_new_mexico.pt
#
# Notes:
#   - Kaggle splits randomly (no subject IDs in that dataset)
#   - ALAMEDA splits by subject_id to avoid data leakage
#   - PADS + New Mexico configs are placeholders until data is ready
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Dataset configs ───────────────────────────────────────
DATASETS = {
    "kaggle": {
        "path":       "data/external/kaggle_mpu9250/Dataset.csv",
        "label_col":  "Result",
        "drop_cols":  [],
        "meta_cols":  [],
        "split_by":   "random",  # no subject_id in this dataset
        "input_dim":  6,         # aX aY aZ gX gY gZ (mag dropped)
        "bottleneck": 8,
        "save_as":    "models/autoencoder_kaggle.pt",
    },
    "alameda": {
        "path":       "data/external/zenodo_alameda/ALAMEDA_PD_tremor_dataset.csv",
        "label_col":  ["Constancy_of_rest", "Kinetic_tremor", "Postural_tremor", "Rest_tremor"],
        "drop_cols":  [],
        "meta_cols":  ["subject_id", "start_timestamp", "end_timestamp"],
        "split_by":   "subject_id",
        "input_dim":  92,
        "bottleneck": 16,
        "save_as":    "models/autoencoder_alameda.pt",
    },
    "pads": {
        "path":       "data/raw/pads-parkinsons-disease-smartwatch-dataset-1.0.0/",
        "label_col":  None,       # unlabeled — autoencoder pretrain only
        "drop_cols":  [],
        "meta_cols":  [],
        "split_by":   "subject_id",
        "input_dim":  None,       # set automatically when loaded
        "bottleneck": 16,
        "save_as":    "models/autoencoder_pads.pt",
    },
    "new_mexico": {
        "path":       "data/external/new_mexico/",  # placeholder until Daniel sends it
        "label_col":  None,
        "drop_cols":  [],
        "meta_cols":  [],
        "split_by":   "random",
        "input_dim":  None,
        "bottleneck": 16,
        "save_as":    "models/autoencoder_new_mexico.pt",
    },
}

# ── Autoencoder ───────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck):
        super().__init__()
        h1 = max(bottleneck * 4, 32)
        h2 = max(bottleneck * 2, 16)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.ReLU(),
            nn.Linear(h2, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2), nn.ReLU(),
            nn.Linear(h2, h1),         nn.ReLU(),
            nn.Linear(h1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Load + prep data ──────────────────────────────────────
def load_data(cfg):
    df = pd.read_csv(cfg["path"])

    # Drop mag cols for kaggle
    mag_cols = [c for c in df.columns if any(m in c.lower() for m in ["magx", "magy", "magz"])]
    drop     = mag_cols + cfg["drop_cols"]

    # Figure out what to exclude from features
    label_col = cfg["label_col"]
    meta_cols = cfg["meta_cols"]

    if isinstance(label_col, list):
        exclude = drop + label_col + meta_cols
    elif label_col:
        exclude = drop + [label_col] + meta_cols
    else:
        exclude = drop + meta_cols

    X = df.drop(columns=exclude, errors="ignore").values.astype(np.float32)

    # Binary label
    if isinstance(label_col, list):
        y = (df[label_col].max(axis=1) > 0).astype(int).values
    elif label_col:
        y = df[label_col].values
    else:
        y = np.zeros(len(X))  # unlabeled

    return df, X, y

# ── Split ─────────────────────────────────────────────────
def split_data(df, X, cfg):
    np.random.seed(42)

    if cfg["split_by"] == "subject_id":
        subjects = df["subject_id"].unique()
        np.random.shuffle(subjects)
        n = len(subjects)
        train_s = subjects[:int(n * 0.70)]
        val_s   = subjects[int(n * 0.70):int(n * 0.85)]
        test_s  = subjects[int(n * 0.85):]
        train_mask = df["subject_id"].isin(train_s).values
        val_mask   = df["subject_id"].isin(val_s).values
        test_mask  = df["subject_id"].isin(test_s).values
        print(f"Subjects — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")
    else:
        idx = np.random.permutation(len(X))
        t, v = int(len(X) * 0.70), int(len(X) * 0.85)
        train_mask = np.zeros(len(X), dtype=bool)
        val_mask   = np.zeros(len(X), dtype=bool)
        test_mask  = np.zeros(len(X), dtype=bool)
        train_mask[idx[:t]]  = True
        val_mask[idx[t:v]]   = True
        test_mask[idx[v:]]   = True

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    print(f"Windows  — train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")
    return X_train, X_val, X_test

# ── Train ─────────────────────────────────────────────────
def train(X_train, X_val, input_dim, bottleneck, save_as, epochs=30):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val)),   batch_size=64)

    model     = Autoencoder(input_dim, bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                val_loss += loss_fn(model(batch), batch).item()

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), save_as)
    print(f"Model saved to {save_as}")

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys(), help="which dataset to train on")
    parser.add_argument("--epochs",  type=int, default=30)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    print(f"\nLoading {args.dataset} from {cfg['path']}")

    df, X, y    = load_data(cfg)
    X_train, X_val, _ = split_data(df, X, cfg)

    input_dim = X_train.shape[1]
    print(f"Input dim: {input_dim}, Bottleneck: {cfg['bottleneck']}")

    train(X_train, X_val, input_dim, cfg["bottleneck"], cfg["save_as"], epochs=args.epochs)