from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data'
SRC_DIR = BASE_DIR / 'src'
NOTEBOOKS_DIR = BASE_DIR / 'notebook'

RAW_DATA_DIR = DATA_DIR / 'raw'
PARKINSON_AT_HOME_DIR = RAW_DATA_DIR / 'parkinson_at_home'
SENSOR_DATA = PARKINSON_AT_HOME_DIR / 'sensor_data'
VIDEO_ANNOTATIONS = PARKINSON_AT_HOME_DIR / 'video_annotations'
