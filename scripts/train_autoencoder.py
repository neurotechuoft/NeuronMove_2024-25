import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv("data/external/kaggle_mpu9250/Dataset.csv")

# Drop mag columns (they're all -1, useless)
mag_cols = [c for c in df.columns if "mag" in c.lower() or "MagX" in c or "MagY" in c or "MagZ" in c]
df = df.drop(columns=mag_cols)

# Separate features and labels
X = df.drop(columns=["Result"]).values.astype(np.float32)
y = df["Result"].values

# ── 2. Split + scale ───────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Convert to tensors
train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_val)),   batch_size=64)

# ── 3. Autoencoder model ───────────────────────────────────
input_dim = X_train.shape[1]

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)   # bottleneck — compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # autoencoder tries to reconstruct input, so we measure reconstruction error

# ── 4. Training loop ───────────────────────────────────────
epochs = 30

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (batch,) in val_loader:
            out = model(batch)
            val_loss += loss_fn(out, batch).item()

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# ── 5. Save model ──────────────────────────────────────────
torch.save(model.state_dict(), "models/autoencoder_kaggle.pt")
print("Model saved!")