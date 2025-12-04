# app.py - Main Flask Application
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# Load the trained model and preprocessing objects
try:
    model = joblib.load('best_heart_disease_model.pkl')
    feature_cols = joblib.load('feature_columns_hd.pkl')
    label_encoders = joblib.load('label_encoders_hd.pkl')
    scaler = joblib.load('scaler_hd.pkl')
    numerical_cols = joblib.load('numerical_columns_hd.pkl')
    print("All model files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Heart Disease Prediction API is running!',
        'model_loaded': model is not None,
        'endpoints': {
            'predict': '/predict [POST]',
            'health': '/ [GET]',
            'model_info': '/info [GET]'
        }
    }), 200

@app.route('/info', methods=['GET'])
def model_info():
    """Get information about the model and required features"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get categorical columns
    categorical_cols = [col for col in feature_cols if col in label_encoders.keys()]
    
    # Build feature info with expected values
    feature_info = {}
    for col in feature_cols:
        if col in label_encoders:
            feature_info[col] = {
                'type': 'categorical',
                'expected_values': label_encoders[col].classes_.tolist()
            }
        else:
            feature_info[col] = {
                'type': 'numerical',
                'description': 'Numeric value'
            }
    
    return jsonify({
        'model_type': type(model).__name__,
        'features': feature_cols,
        'feature_details': feature_info,
        'total_features': len(feature_cols)
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict heart disease based on patient data
    
    Expected JSON format:
    {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up"
    }
    """
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate that all required features are present
        missing_features = [col for col in feature_cols if col not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features,
                'required_features': feature_cols
            }), 400
        
        # Create DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Ensure columns are in the correct order
        input_df = input_df[feature_cols]
        
        # Apply label encoding to categorical features
        for col in label_encoders.keys():
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid value for {col}',
                        'expected_values': label_encoders[col].classes_.tolist(),
                        'received_value': data[col]
                    }), 400
        
        # Apply scaling to numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probability if available
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            probability = {
                'no_disease': float(prediction_proba[0]),
                'disease': float(prediction_proba[1])
            }
        except AttributeError:
            # Some models don't have predict_proba
            probability = None
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'risk_level': 'High' if prediction == 1 else 'Low',
            'input_data': data
        }
        
        if probability:
            response['probability'] = probability
            response['confidence'] = float(max(prediction_proba))
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict heart disease for multiple patients
    
    Expected JSON format:
    {
        "patients": [
            {...patient1_data...},
            {...patient2_data...}
        ]
    }
    """
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'No patients data provided'}), 400
        
        patients = data['patients']
        
        if not isinstance(patients, list) or len(patients) == 0:
            return jsonify({'error': 'patients must be a non-empty list'}), 400
        
        results = []
        
        for idx, patient_data in enumerate(patients):
            try:
                # Create DataFrame
                input_df = pd.DataFrame([patient_data])
                input_df = input_df[feature_cols]
                
                # Encode and scale
                for col in label_encoders.keys():
                    if col in input_df.columns:
                        input_df[col] = label_encoders[col].transform(input_df[col])
                
                input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
                
                # Predict
                prediction = model.predict(input_df)[0]
                
                try:
                    prediction_proba = model.predict_proba(input_df)[0]
                    probability = {
                        'no_disease': float(prediction_proba[0]),
                        'disease': float(prediction_proba[1])
                    }
                except AttributeError:
                    probability = None
                
                result = {
                    'patient_index': idx,
                    'prediction': int(prediction),
                    'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
                    'input_data': patient_data
                }
                
                if probability:
                    result['probability'] = probability
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'patient_index': idx,
                    'error': str(e),
                    'input_data': patient_data
                })
        
        return jsonify({
            'total_patients': len(patients),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5001)