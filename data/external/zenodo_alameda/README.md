# Zenodo — ALAMEDA Parkinson’s tremor (window features)

**Record:** https://zenodo.org/records/10782573  
**DOI:** [10.5281/zenodo.10782573](https://doi.org/10.5281/zenodo.10782573)  
**License:** **CC BY 4.0** (check the Zenodo page if they ever change it)

## What to put here

**File present:** `ALAMEDA_PD_tremor_dataset.csv` (≈5.8 MB on Zenodo) — **~4152 data rows** + header. Git won’t track `.csv` by default; keep a local copy here after download.

## What’s in the CSV (from Zenodo description)

- **99 columns** total: `start_timestamp`, `end_timestamp`, `subject_id`, then **92 feature columns** (time + freq stuff from triaxial accel after preprocessing), then **4 tremor label columns** — each is binary **0/1** derived from MDS-UPDRS III:
  - `Constancy_of_rest`, `Kinetic_tremor`, `Postural_tremor`, `Rest_tremor`
- This is **not raw IMU** — it’s **already featurized windows** (GENEActiv wrist, in-clinic MDS-UPDRS windows; they band-passed etc. per the record page).

## Modeling notes

- Pick **one** label column as target per experiment (or multilabel — up to the team).
- Split by **`subject_id`** when you can so you don’t leak the same person across train/test.
