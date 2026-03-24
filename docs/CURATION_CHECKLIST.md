# Curation todo

Goes with `DATA_SOURCES.md` and the manifest csv. Mostly spreadsheet + asking people where files live, not writing model code.

---

**Done (inventory + docs)**

- [x] `DATA_SOURCES.md` — all team-listed datasets + links + label semantics
- [x] `DATA_MANIFEST_TEMPLATE.csv` — one row per source with paths/notes
- [x] **Kaggle** — `data/external/kaggle_mpu9250/` + CSV + README
- [x] **Zenodo ALAMEDA** — `data/external/zenodo_alameda/` + CSV + README
- [x] **Parkinson@Home** — [Drive link](https://drive.google.com/drive/folders/1XlzJAWlcK7XAoa4dr-TLPmwOjOw35bR4) in docs + manifest
- [x] **IEEE PD-BioStampRC21** — `data/external/ieee_biostamp_rc21/README.md` (download optional; huge)
- [x] **jiehu01 GitHub** — `data/external/github_jiehu01_tremor/README.md` (clone optional)
- [x] **New Mexico** — `data/external/new_mexico/README.md` placeholder until path exists

---

**PADS / repo**

- [ ] If `preprocessed/stratified_subset_file_list.csv` exists locally, count rows → put in manifest **PADS** row
- [ ] If not, run PADS preprocessing when the team is ready (not a curation-only task)

---

**Still blocking on others**

- [ ] **New Mexico** — link or folder from team → update README + manifest (`url_or_drive_note` + `local_path`)

---

**Handoff**

- [ ] Tell whoever’s training: read `DATA_SOURCES.md` + `data/external/*/README.md`; CSVs under `data/external/` are gitignored
- [ ] Optional: copy `DATA_MANIFEST_TEMPLATE.csv` → `DATA_MANIFEST.csv` when the team freezes the sheet

---

**Reminder:** split by **subject / participant id** when one person has many windows — not random rows.
