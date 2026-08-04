import sys
import os

# Add root folder to sys.path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessor import download_dataset, preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def run_pipeline():
    print("=" * 60)
    print("STEP 1: Downloading & Preprocessing Data")
    print("=" * 60)
    download_dataset()
    preprocess_data()
    
    print("\n" + "=" * 60)
    print("STEP 2: Training Logistic Regression Model")
    print("=" * 60)
    train_model()
    
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating Model & Generating Visuals")
    print("=" * 60)
    evaluate_model()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline()
