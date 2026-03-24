# Curation todo

Goes with `DATA_SOURCES.md` and the manifest csv. Mostly spreadsheet + asking people where files live, not writing model code.

---

**Already started**

- [x] DATA_SOURCES rough draft
- [x] manifest template csv
- [x] this checklist

---

**PADS / repo (can do without new downloads)**

- [ ] check if `preprocessed/` actually exists on your laptop (sometimes people never ran preprocessing)
- [ ] if `stratified_subset_file_list.csv` exists, count roughly how many patients / rows → put in manifest instead of TBD
- [ ] sampling rate is 100hz and window length 976 in baseline script — already noted in DATA_SOURCES

**Drive**

- [ ] get parkinson’s @ home folder link from whoever has it on drive
- [ ] paste link or path into manifest `url_or_drive_note` / `local_path` when you have a copy

**External stuff (when you have time)**

Rough order that’s usually less painful:

1. ~~kaggle MPU9250~~ — local: `data/external/kaggle_mpu9250/` (see README there)
2. zenodo alameda — one csv
3. ieee biostamp — dataport login, remember subject-level only
4. github clone — list what’s inside before mixing with other data

**Handoff**

- [ ] when manifest is mostly filled, rename copy to `DATA_MANIFEST.csv` or whatever team agrees on
- [ ] tell whoever’s training: paths + what’s labeled how

---

**Reminder for everyone:** if data has multiple rows per person, split by **subject id** not random rows. Otherwise metrics lie.
