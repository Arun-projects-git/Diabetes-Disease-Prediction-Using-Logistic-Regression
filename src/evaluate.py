import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

# Set non-interactive backend for matplotlib to avoid GUI window popup during pipeline run
plt.switch_backend('Agg')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'plots')

def evaluate_model():
    # Load test data
    test_path = os.path.join(PREPROCESSED_DIR, 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Please run preprocessing first.")
        
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=['Outcome'])
    y_test = test_df['Outcome']
    
    # Load model and scaler
    model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Please train the model first.")
        
    print("Loading model and scaler...")
    with open(model_path, 'wb' if not os.path.exists(model_path) else 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'wb' if not os.path.exists(scaler_path) else 'rb') as f:
        scaler = pickle.load(f)
        
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Predict
    y_pred = model.predict(X_test_scaled_df)
    y_prob = model.predict_proba(X_test_scaled_df)[:, 1]
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Save metrics to JSON
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4)
    }
    
    metrics_path = os.path.join(PLOTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Generate Confusion Matrix Plot
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Non-Diabetic', 'Diabetic'],
        yticklabels=['Non-Diabetic', 'Diabetic']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Generate ROC Curve Plot
    print("Generating ROC Curve plot...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Generate Feature Importance / Coefficients Plot
    print("Generating Feature Coefficients plot...")
    coefficients = model.coef_[0]
    features = X_test.columns
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'Absolute Coefficient': np.abs(coefficients)
    }).sort_values(by='Absolute Coefficient', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x='Coefficient', y='Feature', data=coef_df,
        palette='viridis', hue='Feature', legend=False
    )
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title('Logistic Regression Feature Coefficients')
    plt.xlabel('Coefficient Value (Impact on Prediction)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_coefficients.png'), dpi=300)
    plt.close()
    
    print("All plots generated and saved in:", PLOTS_DIR)

if __name__ == '__main__':
    evaluate_model()
