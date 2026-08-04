import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'diabetes.csv')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')

DATASET_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

def download_dataset():
    """Downloads the PIMA Indians Diabetes Dataset if it doesn't exist."""
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        with open(RAW_DATA_PATH, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")

def preprocess_data():
    """Loads, cleans, splits and prepares the dataset."""
    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    
    # In Pima Indians dataset, 0 is invalid for: Glucose, BloodPressure, SkinThickness, Insulin, BMI
    zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("Handling missing values (replacing 0 with NaN for: %s)..." % ", ".join(zero_fields))
    for field in zero_fields:
        df[field] = df[field].replace(0, np.nan)
        
    # Split into features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Split into train and test sets (80% train, 20% test)
    print("Splitting dataset into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Impute missing values using training set medians (to avoid data leakage)
    print("Imputing missing values with training set medians...")
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    
    # Save the medians for inference imputation
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    medians.to_json(os.path.join(PREPROCESSED_DIR, 'imputation_medians.json'))
    print("Imputation medians saved.")
    
    # Save training and test sets
    train_df = X_train.copy()
    train_df['Outcome'] = y_train
    
    test_df = X_test.copy()
    test_df['Outcome'] = y_test
    
    train_df.to_csv(os.path.join(PREPROCESSED_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(PREPROCESSED_DIR, 'test.csv'), index=False)
    
    print("Preprocessing completed. Files saved in:", PREPROCESSED_DIR)

if __name__ == '__main__':
    download_dataset()
    preprocess_data()
