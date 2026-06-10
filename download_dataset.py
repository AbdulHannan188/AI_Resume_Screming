"""
Dataset Download Helper
=======================
Downloads the "UpdatedResumeDataSet" from Kaggle and places it in data/.

Prerequisites
-------------
1. Install the Kaggle CLI:   pip install kaggle
2. Create a Kaggle account at https://www.kaggle.com
3. Go to  Account -> Settings -> API -> Create New Token
   This downloads  kaggle.json  — move it to  ~/.kaggle/kaggle.json
   (Linux/Mac) or  C:/Users/<user>/.kaggle/kaggle.json  (Windows)
4. Set permissions (Linux/Mac only):  chmod 600 ~/.kaggle/kaggle.json

Then run:
    python download_dataset.py

Dataset
-------
Name   : UpdatedResumeDataSet
Source : https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
Rows   : ~2484
Cols   : Category, Resume
"""

import os
import sys
import subprocess
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATASET_SLUG = 'snehaanbhawal/resume-dataset'
CSV_NAME = 'UpdatedResumeDataSet.csv'


def check_kaggle_credentials():
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print('\n  kaggle.json not found at ~/.kaggle/kaggle.json')
        print('  Steps to fix:')
        print('    1. Go to https://www.kaggle.com/settings')
        print('    2. Scroll to "API" section -> "Create New Token"')
        print('    3. Move the downloaded kaggle.json to ~/.kaggle/')
        print('    4. Run: chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac)')
        return False
    return True


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    target = os.path.join(DATA_DIR, CSV_NAME)

    if os.path.exists(target):
        print(f'  Dataset already exists at {target}')
        return True

    if not check_kaggle_credentials():
        return False

    print(f'  Downloading dataset: {DATASET_SLUG}')
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', DATASET_SLUG,
             '-p', DATA_DIR, '--unzip'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f'  Kaggle CLI error:\n{result.stderr}')
            return False
        print(f'  Downloaded successfully to {DATA_DIR}/')

        # Rename if needed
        for fname in os.listdir(DATA_DIR):
            if fname.endswith('.csv') and fname != CSV_NAME:
                os.rename(os.path.join(DATA_DIR, fname), target)
                print(f'  Renamed {fname} -> {CSV_NAME}')
                break

        return os.path.exists(target)
    except FileNotFoundError:
        print('  kaggle CLI not found. Install it: pip install kaggle')
        return False
    except subprocess.TimeoutExpired:
        print('  Download timed out.')
        return False


if __name__ == '__main__':
    print('=' * 55)
    print('  AI Resume Screening — Dataset Download')
    print('=' * 55)
    success = download_dataset()
    if success:
        import pandas as pd
        df = pd.read_csv(os.path.join(DATA_DIR, CSV_NAME))
        print(f'\n  Dataset info:')
        print(f'    Rows       : {len(df)}')
        print(f'    Columns    : {list(df.columns)}')
        print(f'    Categories : {df.iloc[:, 0].nunique()}')
        print('\n  Ready to run: python main.py')
    else:
        print('\n  Download failed. The project will use built-in sample data.')
        print('  Run python main.py to proceed with the sample dataset.')
