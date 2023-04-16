import os
from pathlib import Path
from time import time

REPO_DIR = Path(__file__).resolve().parent
DATASETS_DIR = REPO_DIR / 'datasets'
MODEL_DIR = REPO_DIR / 'models'
REPORT_DIR = REPO_DIR / 'reports'
CURRRENT_TIME = f'local_{int(time() * 1000)}'

def get_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    return MODEL_DIR

def get_report_dir():
    if not os.path.exists(REPORT_DIR):
        os.mkdir(REPORT_DIR)
    if not os.path.exists(REPORT_DIR / CURRRENT_TIME):
        os.mkdir(REPORT_DIR / CURRRENT_TIME)
    return REPORT_DIR / CURRRENT_TIME