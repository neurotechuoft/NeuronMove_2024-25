# Parkinson@Home - IMU sensor data and video annotations for real-life tremor detection

For any questions related to the data repository or Git repository, please contact [Nienke Timmermans](mailto:nienke.timmermans@radboudumc.nl).


This data repository contains four distinct directories:
1. docs
2. sensor_data
3. video_annotations
4. clinical_data

### docs
The study protocol (_study_protocol.pdf_) and video annotation protocol (_video_annotation_protol.pdf_) are contained in this directory. Any comments on or deviations from these protocols were registered separately (_comments_and_protocol_deviations.xlsx_).

#### 2. sensor_data
This directory contains the raw sensor data  of the Parkinson cohort (_phys_cur_PD_merged.mat_) and the control cohort (_phys_cur_HC_merged.mat_) which can be read into MATLAB. Each row corresponds to a single participant, identified by the prefix _hbv_ with subsequent participant id (e.g., _hbv002_) in the first column _id_. The second and third columns, _LW_ and _RW_, contain sensor data of the left and right wrist respectively. Both accelerometer and gyroscope data are contained. The final columns _peakstart_ and _peakend_ represent the start and end time of the recording, allowing the sensor to be matched with the video recordings.  

#### 3. video_annotations
We used ELAN 6.4 for annotating the video recordings corresponding to the collected sensor data. For privacy reasons, we have excluded the videos of the participants and stored the video annotations of tremor and activities. The video annotations are stored in _labels_PD_phys_tremorfog_sharing.mat_ for the Parkinson cohort and in _labels_HC_phys_sharing.mat_ for the control cohort. Again, each row corresponds to a single participant, identified by the prefix _hbv_ with subsequent participant id. The physical activity labels are stored in the columns _premed_ and _postmed_ for the Parkinson cohort and _pre_ and _post_ for the control cohort. 
The tremor labels are stored in the columns _premed_tremor_ and _postmed_tremor_. All columns that end with _start or _end indicate the start or end time of the video recordings, so that it can be synchronized with the sensor data.

#### 4. clinical_data
The clinical data of the participants was stripped of privacy-sensitive data and stored in this directory. The file _patient_info.csv_ contains demographic and UPDRS data per participant. 