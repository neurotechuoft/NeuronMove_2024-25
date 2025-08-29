# NeurotechUofT - Software

**NeuronMove** is a project under **NeuroTech UofT** that aims to predict Parkinson's hand tremors using physiological signals, primarily EEG and accelerometer data. The end goal is to integrate this system with real-time hardware to help suppress or mitigate tremors in PD patients.

This repository contains the codebase for the 2024–25 team’s efforts, which focus on signal preprocessing, feature extraction, and future integration with deep learning models.

---

## Project Overview

This repository focuses on building the software pipeline needed to achieve this goal. It includes:

- Preprocessing raw data from public datasets

- Extracting tremor-related events using accelerometer signals

- Aligning and segmenting EEG data based on those events

- Preparing the data for machine learning and deep learning models

Our initial focus is on data preprocessing and validation, with deep learning integration planned in the next development phase.

---

## Preprocessing Pipeline

The preprocessing pipeline prepares raw EEG and accelerometer recordings for use in machine learning models.

### 1. **Accelerometer-Based Label Extraction**

Since the dataset does not contain ground-truth tremor labels, we derive **pre-tremor, tremor, and non-tremor** labels from the accelerometer signals using the method described in [PMC10668446](https://pmc.ncbi.nlm.nih.gov/articles/PMC10668446/) which is summarized in the image below.

<img width="1578" height="1014" alt="image" src="https://github.com/user-attachments/assets/0033db3e-c63a-4314-ad96-1428f9933aa0" />


These extracted timestamps are then used to label both EEG and accelerometer segments for training and evaluation. The labeling implementation can be found in:
src/accelerometer/_02_preprocess.py


### 2. **EEG Preprocessing**

Raw EEG signals are initially stored in `.mat` files in (from EEGLAB) and must be converted to the MNE-compatible `.fif` format for further analysis. This conversion and preprocessing pipeline includes:

- **Loading and parsing EEGLAB `.mat` files**, including signal data, sampling rates, and channel labels  
- **Identifying and labeling channel types** (EEG, EOG, accelerometer, stimulation, etc.)
- **Setting standard electrode montage** for spatial mapping
- **Converting EEGLAB events** (e.g., stimulus triggers) into MNE-style annotations
- **Saving processed files** in `.fif` format for use in later stages

This prepares the EEG data for downstream analysis and ensures compatibility with MNE tools.

You can find this conversion logic in `src/eeg/_10_convert_mat_to_fif.py`

---

