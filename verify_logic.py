import joblib
import pandas as pd
import numpy as np
import os

def verify_app_logic():
    print("Verifying App Logic...")
    
    # Check files
    if not os.path.exists('best_model.pkl'):
        print("ERROR: best_model.pkl not found!")
        return
    if not os.path.exists('scaler.pkl'):
        print("ERROR: scaler.pkl not found!")
        return
        
    # 1. Load Resources
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("SUCCESS: Model and Scaler loaded.")
    except Exception as e:
        print(f"ERROR: Failed to load resources: {e}")
        return

    # 2. Create Sample Data (Simulating App Input)
    # Feature Names: 'UDI', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'Type_H', 'Type_L', 'Type_M'
    
    input_data = pd.DataFrame([{
        'UDI': 0, 
        'Air_temperature_K': 300.0,
        'Process_temperature_K': 310.0,
        'Rotational_speed_rpm': 1500,
        'Torque_Nm': 40.0,
        'Tool_wear_min': 0,
        'Type_H': 0,
        'Type_L': 1,
        'Type_M': 0
    }])
    
    print(f"Input Data:\n{input_data}")
    
    # 3. Predict
    try:
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        print(f"SUCCESS: Prediction Run.")
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability:.4f}")
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")

if __name__ == "__main__":
    verify_app_logic()
