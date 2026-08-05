import os
import psycopg2
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)   
DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'plots')
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'preprocessed')

model = None
scaler = None
imputation_medians = None

def load_resources():
    global model, scaler, imputation_medians
    model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    medians_path = os.path.join(PREPROCESSED_DIR, 'imputation_medians.json')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Model and Scaler loaded successfully.")
    else:
        print("WARNING: Model or Scaler not found! Please run the training pipeline first.")
        
    if os.path.exists(medians_path):
        with open(medians_path, 'r') as f:
            imputation_medians = json.load(f)
        print("Imputation medians loaded successfully.")
    else:
        print("WARNING: Imputation medians not found!")

# Try loading at startup
load_resources()

@app.route('/')
def home():
    # Render main dashboard page
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_path = os.path.join(PLOTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return jsonify({'status': 'success', 'data': metrics})
    else:
        return jsonify({'status': 'error', 'message': 'Metrics not generated yet. Run pipeline.'}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    global model, scaler, imputation_medians
    
    # Reload model if they weren't loaded at startup (e.g. if pipeline ran after app started)
    if model is None or scaler is None or imputation_medians is None:
        load_resources()
        if model is None or scaler is None:
            return jsonify({'status': 'error', 'message': 'Prediction model is not available. Please run the training pipeline.'}), 500
            
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
            
        # Feature names in order
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Parse inputs
        input_data = {}
        for feature in feature_names:
            val = data.get(feature)
            
            # Treat empty strings or None as missing
            if val is None or val == '':
                # Impute missing input using median
                input_data[feature] = float(imputation_medians.get(feature, 0.0))
            else:
                # If feature is one of zero_fields and value is 0, treat as missing and impute
                val_float = float(val)
                zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
                if feature in zero_fields and val_float == 0.0:
                    input_data[feature] = float(imputation_medians.get(feature, 0.0))
                else:
                    input_data[feature] = val_float

        # Convert to pandas DataFrame with column names (so scaler and model preserve name matching)
        input_df = pd.DataFrame([input_data])
        
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Predict class and probability
        prediction = int(model.predict(input_scaled_df)[0])
        probability = float(model.predict_proba(input_scaled_df)[0][1])
        
        # Generate personalized clinical recommendations
        recommendations = []
        
        # Check Glucose
        glucose_val = input_data['Glucose']
        if glucose_val >= 140:
            recommendations.append("Your glucose level is high. Consider monitoring carbohydrate intake and consulting a healthcare professional for a glucose tolerance test.")
        elif glucose_val < 70:
            recommendations.append("Your glucose level is low (hypoglycemia risk). Ensure adequate nutrition and consult a doctor if you feel dizzy or fatigued.")
            
        # Check BloodPressure
        bp_val = input_data['BloodPressure']
        if bp_val >= 90:
            recommendations.append("Your blood pressure is in the hypertensive range. Consider reducing sodium intake and managing stress.")
        elif bp_val < 60:
            recommendations.append("Your blood pressure is relatively low. Monitor for symptoms like dizziness.")
            
        # Check BMI
        bmi_val = input_data['BMI']
        if bmi_val >= 30:
            recommendations.append("Your BMI indicates obesity, which is a significant risk factor for diabetes. regular exercise and a balanced weight management plan are recommended.")
        elif bmi_val >= 25:
            recommendations.append("Your BMI indicates you are overweight. A balanced diet and daily active exercise can help lower risk factors.")
            
        # Check Age
        age_val = input_data['Age']
        if age_val >= 45:
            recommendations.append("At age 45 or older, the risk of developing diabetes increases. Regular screening is advised.")
            
        # Default recommendation if outcome is high risk
        if prediction == 1:
            recommendations.append("Based on the input features, you are classified in the HIGH RISK category. We strongly advise schedule a clinical evaluation with a physician.")
        else:
            recommendations.append("Your current profile suggests a LOW RISK of diabetes. Maintain a healthy lifestyle with proper diet and active physical exercise.")
            
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probability': probability,
            'recommendations': recommendations,
            'imputed_values': {
                k: round(v, 2) for k, v in input_data.items() if (data.get(k) is None or data.get(k) == '' or (k in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and float(data.get(k)) == 0))
            }
        })
        
    except ValueError as ve:
        return jsonify({'status': 'error', 'message': f'Invalid input value: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
