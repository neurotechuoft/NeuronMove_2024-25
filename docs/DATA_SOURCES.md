# Where our data lives (curation notes)

Last updated 2026-03-24

Trying to keep one doc so we’re not all asking “wait is this labeled per window or per person?”

---

**Quick vocab**

- **Sample/window labels** — each row or segment has a label. Fine for normal supervised training.
- **Subject-level only** — you only know “this person is PD vs healthy”, not every window. Don’t slap that label on every row unless we actually mean to (leakage / wrong task).
- **Unlabeled** — fine for autoencoder / SSL pretrain, no labels needed.

---

## Stuff that’s supposed to be good for supervised (labeled-ish)

| What | Link / where | Notes |
|------|----------------|-------|
| Parkinson’s @ Home | Drive — **path still needed** | IMU + annotations; copy lives on team drive |
| Kaggle MPU9250 tremor | https://www.kaggle.com/datasets/aaryapandya/hand-tremor-dataset-collected-using-mpu9250-sensor | Local copy: `data/external/kaggle_mpu9250/Dataset.csv` + README there. **3 MPU9250** on hand; `Result`: **1 = shaking**, **0 = stable**; ~28k rows; **mag = -1** (use acc+gyro); **License: Unknown** on Kaggle |
| Zenodo ALAMEDA | https://zenodo.org/records/10782573 | CSV with 92 features per window + tremor labels — **not** raw IMU in that file. not fully vetted yet |

---

## Unlabeled pool / pretrain only / use carefully

| What | Link / where | Notes |
|------|----------------|-------|
| PADS | `data/raw/pads-...` and `preprocessed/` in this repo | we already have scripts + baseline RF on stratified labels; raw can still be used as unlabeled for SSL if we want |
| New Mexico | ??? | brought up in team sync — **still need where it actually is** |
| PD-BioStampRC21 | https://ieee-dataport.org/open-access/pd-biostamprc21-parkinsons-disease-accelerometry-dataset-five-wearable-sensor-study-0 | PD vs HC at subject level — ieee dataport account |
| jiehu01 github tremor repo | https://github.com/jiehu01/Parkinson-s-Disease-Tremor-Dataset | bunch of IMU datasets, severity 0–3 in readme — double-check each sub-dataset before mixing |

---

## Already in the repo (paths)

- `preprocessed/movement/*_ml.bin` + `stratified_subset_file_list.csv` — what the baseline uses
- `data/raw/parkinson_at_home/README.md` — explains the .mat layout (sensor_data, video_annotations, etc.). big files might not be in git, only readme
- `data/raw/pads-parkinsons-disease-smartwatch-dataset-1.0.0/` — PADS + their preprocessing scripts
- baseline script: `scripts/train_baseline_model.py` — 100 Hz, 976 samples per window

Extra: `DATA_MANIFEST_TEMPLATE.csv` is the spreadsheet version of this; fill paths as we go.

---

## Things to ask the team

- Exact **Drive folder** for Parkinson’s @ Home
- **New Mexico** — link or folder?
- Are we ok merging ALAMEDA with other stuff license-wise?
- Splits should be **by patient** where one person has many windows (PADS, biostamp, etc.) — not random rows

---

## todo (curation)

- [ ] drive paths + new mexico when people reply
- [ ] fill manifest when something gets downloaded
- [ ] ping whoever’s training when manifest isn’t all TBD anymore
