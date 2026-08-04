import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def train_model():
    # Load training data
    train_path = os.path.join(PREPROCESSED_DIR, 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Please run preprocessing first.")
        
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=['Outcome'])
    y_train = train_df['Outcome']
    
    # Scale features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Convert scaled features back to DataFrame to preserve column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    # Define hyperparameter grid for Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # 'liblinear' works well with l1 and l2 penalties on small datasets
    }
    
    print("Training Logistic Regression model with GridSearchCV...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled_df, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Save the model, scaler, and feature names
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == '__main__':
    train_model()
