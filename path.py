import os
from pathlib import Path
from time import time

REPO_DIR = Path(__file__).resolve().parent
DATASETS_DIR = '/nfs/hpc/share/yonge/data/cs579'
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

def check_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_mnist_label_dir(dir):
    for i in range(10):
        os.mkdir(f'{dir}/{i}')