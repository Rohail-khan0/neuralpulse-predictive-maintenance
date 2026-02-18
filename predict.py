import joblib
import pandas as pd
import numpy as np

# Load saved model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load feature names to get exact column order
with open('feature_names.txt', 'r') as f:
    expected_features = [line.strip() for line in f.readlines()]

print(f"Expected features: {expected_features}")

def predict_machine_failure(sensor_data):
    """
    Predict machine failure based on sensor data.
    
    Parameters:
    -----------
    sensor_data : dict or pd.DataFrame
        Sensor readings - accepts both old format (with brackets) and new format (with underscores)
        
    Returns:
    --------
    dict : Prediction results including probability and recommendation
    """
    # Convert to DataFrame if dict
    if isinstance(sensor_data, dict):
        sensor_data = pd.DataFrame([sensor_data])
    
    # Clean column names - convert brackets/spaces to underscores
    sensor_data.columns = sensor_data.columns.str.replace('[', '', regex=False)
    sensor_data.columns = sensor_data.columns.str.replace(']', '', regex=False)
    sensor_data.columns = sensor_data.columns.str.replace('<', '', regex=False)
    sensor_data.columns = sensor_data.columns.str.replace('>', '', regex=False)
    sensor_data.columns = sensor_data.columns.str.replace(' ', '_')
    
    # Add missing columns with default values if needed
    for feature in expected_features:
        if feature not in sensor_data.columns:
            sensor_data[feature] = 0
    
    # Reorder columns to match training order exactly
    sensor_data = sensor_data[expected_features]
    
    # Scale features
    sensor_data_scaled = scaler.transform(sensor_data)
    
    # Make prediction
    prediction = model.predict(sensor_data_scaled)[0]
    probability = model.predict_proba(sensor_data_scaled)[0, 1]
    
    # Generate recommendation
    if prediction == 1:
        if probability >= 0.9:
            risk_level = "🔴 CRITICAL RISK"
            recommendation = "IMMEDIATE ACTION! Stop machine and perform emergency maintenance."
        elif probability >= 0.75:
            risk_level = "🟠 HIGH RISK"
            recommendation = "Schedule maintenance within 2-4 hours."
        elif probability >= 0.6:
            risk_level = "🟡 MEDIUM RISK"
            recommendation = "Schedule maintenance within 8-12 hours."
        else:
            risk_level = "🟡 ELEVATED RISK"
            recommendation = "Schedule maintenance within 24 hours."
    else:
        if probability >= 0.4:
            risk_level = "🟢 LOW-MEDIUM RISK"
            recommendation = "Continue monitoring."
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = "Machine operating normally."
    
    return {
        'failure_predicted': bool(prediction),
        'failure_probability': float(probability),
        'risk_level': risk_level,
        'recommendation': recommendation
    }

if __name__ == "__main__":
    # Example with CORRECT column names (underscores)
    example_data = {
        'UDI': 1,
        'Air_temperature_K': 298.5,
        'Process_temperature_K': 309.0,
        'Rotational_speed_rpm': 1500,
        'Torque_Nm': 45.0,
        'Tool_wear_min': 30,
        'Type_H': 0,
        'Type_L': 1,
        'Type_M': 0
    }
    
    result = predict_machine_failure(example_data)
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Failure Predicted: {result['failure_predicted']}")
    print(f"Failure Probability: {result['failure_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print("="*60)