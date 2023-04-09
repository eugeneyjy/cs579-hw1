import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
DATASETS_DIR = REPO_DIR / 'datasets'
MODEL_DIR = REPO_DIR / 'models'
REPORT_DIR = REPO_DIR / 'reports'

def get_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    return MODEL_DIR

def get_report_dir():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    return REPORT_DIR
