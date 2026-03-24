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
| Parkinson’s @ Home | **[Google Drive folder](https://drive.google.com/drive/folders/1XlzJAWlcK7XAoa4dr-TLPmwOjOw35bR4)** | Same README structure as below; on Drive you also have **`.csv data`** + **`.csv labels`**, **formatted data**, etc. Full `.mat`/CSV on Drive or local copy; repo has `data/raw/parkinson_at_home/README.md` |
| Kaggle MPU9250 tremor | https://www.kaggle.com/datasets/aaryapandya/hand-tremor-dataset-collected-using-mpu9250-sensor | Local copy: `data/external/kaggle_mpu9250/Dataset.csv` + README there. **3 MPU9250** on hand; `Result`: **1 = shaking**, **0 = stable**; ~28k rows; **mag = -1** (use acc+gyro); **License: Unknown** on Kaggle |
| Zenodo ALAMEDA | https://zenodo.org/records/10782573 | Local: `data/external/zenodo_alameda/ALAMEDA_PD_tremor_dataset.csv` + README. **~4152 windows**; **92 feat cols + 4 binary tremor cols**; **CC BY 4.0**; **not raw IMU**; split by **`subject_id`** |

---

## Unlabeled pool + “subject-level only” sources (semi-supervised plan)

What the team called out: **unlabeled** = **New Mexico** + **raw PADS** (paths below).  
Also: **PD vs HC** (or study-level) **not** sample-level labels — **don’t treat every window as a class label** without a real design.

| What | Link / where | Notes |
|------|----------------|-------|
| **New Mexico** | TBD | Unlabeled pool — add link when someone shares it (`data/external/new_mexico/README.md`) |
| **PADS raw** | `data/raw/pads-parkinsons-disease-smartwatch-dataset-1.0.0/` | Raw + scripts in repo; **pretrain without labels** or use stratified labels only when you mean to |
| **PD-BioStampRC21** | [IEEE DataPort](https://ieee-dataport.org/open-access/pd-biostamprc21-parkinsons-disease-accelerometry-dataset-five-wearable-sensor-study-0) | PD vs HC / protocol-level — **not** per-sample tremor labels; **large** download; optional |
| **jiehu01 GitHub** (autoencoder) | [repo](https://github.com/jiehu01/Parkinson-s-Disease-Tremor-Dataset) | Lots of IMU tremor data for **autoencoder**-style pretraining; **cite** per sub-dataset |

**Preprocessed PADS** (`preprocessed/movement/`, stratified CSV) is still the **baseline supervised** path — same `data/raw/pads` tree, different use.

---

## Parkinson@Home layout (from official README)

**Drive:** [Parkinson@Home folder](https://drive.google.com/drive/folders/1XlzJAWlcK7XAoa4dr-TLPmwOjOw35bR4) (Shared with me → Data → …)

Same text as `data/raw/parkinson_at_home/README.md` — four top-level dirs (plus on Drive: **`.csv data`**, **`.csv labels`**, **formatted data**, etc.):

1. **`docs`** — study protocol PDFs, annotation protocol, protocol deviations xlsx  
2. **`sensor_data`** — PD: `phys_cur_PD_merged.mat`, controls: `phys_cur_HC_merged.mat` (MATLAB). Each **row = one participant** (`id` like `hbv002`). **`LW` / `RW`** = left/right wrist accel+gyro; **`peakstart` / `peakend`** = recording window (sync with video).  
3. **`video_annotations`** — PD: `labels_PD_phys_tremorfog_sharing.mat`, HC: `labels_HC_phys_sharing.mat`. Activity cols `premed`/`postmed` (PD) or `pre`/`post` (HC); tremor: `premed_tremor`, `postmed_tremor`. `*_start` / `*_end` = time ranges to align with sensors.  
4. **`clinical_data`** — `patient_info.csv` (demographics + UPDRS; de-identified)

Questions about the **data repository**: contact listed in that README (Radboud).

**For ML:** align sensor rows with annotation time ranges; split by **participant** (`id`), not random rows.

---

## Already in the repo (paths)

- `preprocessed/movement/*_ml.bin` + `stratified_subset_file_list.csv` — what the baseline uses
- `data/raw/parkinson_at_home/README.md` — explains the .mat layout (sensor_data, video_annotations, etc.). big files might not be in git, only readme
- `data/raw/pads-parkinsons-disease-smartwatch-dataset-1.0.0/` — PADS + their preprocessing scripts
- baseline script: `scripts/train_baseline_model.py` — 100 Hz, 976 samples per window

Extra: `DATA_MANIFEST_TEMPLATE.csv` is the spreadsheet version of this; fill paths as we go.

---

## Things to ask the team

- **New Mexico** — link or folder?
- Are we ok merging ALAMEDA with other stuff license-wise?
- Splits should be **by patient** where one person has many windows (PADS, biostamp, etc.) — not random rows

---

## todo (curation)

- [x] Parkinson@Home Drive + Kaggle + Zenodo + links for SSL pool (PADS raw, BioStamp, jiehu01, New Mexico)
- [ ] **New Mexico** — path when someone sends it
- [ ] Ping whoever’s training + rename manifest if the team wants `DATA_MANIFEST.csv`
