from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import hashlib
import json

# --- Import your Blockchain class ---
from blockchain import Block, Blockchain
# -----------------------------------

app = Flask(__name__)

# Define the directory where your model components are saved in Google Drive
MODEL_DIR = '/content/drive/MyDrive/Diabetic_Prediction_Model'
# --- Define the path for your blockchain persistence file ---
BLOCKCHAIN_FILE = os.path.join(MODEL_DIR, 'diabetes_blockchain.json')
# -------------------------------------------------------------

# Load the trained model components
ensemble_model = None
scaler = None
gender_encoder = None
selected_columns = None
original_feature_names = None

try:
    ensemble_model = joblib.load(os.path.join(MODEL_DIR, 'ensemble_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    gender_encoder = joblib.load(os.path.join(MODEL_DIR, 'gender_encoder.pkl'))
    selected_columns = joblib.load(os.path.join(MODEL_DIR, 'selected_columns.pkl'))
    original_feature_names = joblib.load(os.path.join(MODEL_DIR, 'original_feature_names.pkl'))
    print("All model components loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model components: {e}")
    print(f"Please ensure your model files are in {MODEL_DIR} and Drive is mounted.")

# --- Initialize your Blockchain instance ---
d_blockchain = Blockchain()
# Try to load existing chain, otherwise create genesis block
if not d_blockchain.load_chain_from_file(BLOCKCHAIN_FILE):
    d_blockchain.create_genesis_block()
print(f"Blockchain initialized. Total blocks: {len(d_blockchain.chain)}")
# -----------------------------------------

# Mapping for display
PREDICTION_MAP = {0: 'N', 1: 'P', 2: 'Y'}
FULL_PREDICTION_MAP = {'N': 'Non-Diabetic', 'P': 'Pre-Diabetic', 'Y': 'Diabetic'}

# Recommendations based on predicted stage
RECOMMENDATIONS = {
    'N': [
        "Maintain a balanced diet with plenty of fruits, vegetables, and whole grains.",
        "Engage in regular physical activity (at least 30 minutes most days of the week).",
        "Monitor your weight and maintain a healthy BMI.",
        "Stay hydrated and limit sugary drinks.",
        "Get regular check-ups with your doctor."
    ],
    'P': [
        "Cut down significantly on sugar and processed foods.",
        "Increase your exercise and monitor your weight closely.",
        "Check blood sugar levels regularly as advised by a doctor.",
        "Consider consulting a nutritionist for a personalized meal plan.",
        "Reduce stress and ensure adequate sleep."

    ],
    'Y': [
        "Strictly follow your doctor's treatment plan and medication schedule.",
        "Adhere to a diabetic-friendly diet, focusing on low glycemic index foods.",
        "Regularly monitor blood sugar levels and keep a log.",
        "Engage in consistent physical activity, as approved by your doctor.",
        "Attend all scheduled medical appointments and screenings.",
        "Educate yourself on diabetes management and potential complications."
    ]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if ensemble_model is None or scaler is None or gender_encoder is None or \
       selected_columns is None or original_feature_names is None:
        return render_template('index.html', prediction_result="Error", confidence_score="N/A",
                               recommendations=["Model components not loaded. Please check server logs and file paths."])

    try:
        data = request.form.to_dict()
        print(f"DEBUG: Raw form data: {data}")

        input_values = {}
        for key, value in data.items():
            if key in ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']:
                input_values[key] = float(value)
            else:
                input_values[key] = value
        print(f"DEBUG: Processed input_values (after unit conversion fix): {input_values}")

        processed_input = {}
        for feature in original_feature_names:
            if feature == 'Gender':
                gender_val = input_values.get('Gender', 'F')
                processed_input[feature] = gender_encoder.transform([str(gender_val).upper()])[0]
            elif feature in input_values:
                processed_input[feature] = input_values[feature]
            else:
                processed_input[feature] = 0
        print(f"DEBUG: processed_input before DataFrame: {processed_input}")

        input_for_scaling = pd.DataFrame([processed_input], columns=original_feature_names)
        print(f"DEBUG: input_for_scaling DataFrame:\n{input_for_scaling}")

        scaled_input = scaler.transform(input_for_scaling)
        print(f"DEBUG: scaled_input (after scaler.transform):\n{scaled_input}")

        final_input = scaled_input[:, selected_columns]
        print(f"DEBUG: final_input (after feature selection):\n{final_input}")
        print(f"DEBUG FINAL INPUT SHAPE: {final_input.shape}")
        print(f"DEBUG FINAL INPUT SAMPLE: {final_input[0]}")

        prediction_numeric = ensemble_model.predict(final_input)[0]
        prediction_proba = ensemble_model.predict_proba(final_input)[0]

        print(f"DEBUG: Ensemble Model Probabilities (N=0, P=1, Y=2): {prediction_proba}")

        # --- START NEW, AGGRESSIVE QUICK FIX FOR PRE-DIABETIC (P) CLASSIFICATION ---
        hba1c_input = input_values.get('HbA1c')

        HBA1C_PREDIABETES_START = 5.7
        HBA1C_PREDIABETES_END = 6.4

        # If HbA1c is within the *medical* pre-diabetic range,
        # FORCE the prediction to be Pre-Diabetic (P = 1).
        if hba1c_input is not None and HBA1C_PREDIABETES_START <= hba1c_input <= HBA1C_PREDIABETES_END:
            prediction_numeric = 1 # Force to P
            print(f"DEBUG: Aggressive Quick Fix Applied: HbA1c ({hba1c_input}) is in P range. Forced prediction to P.")
        # --- END NEW, AGGRESSIVE QUICK FIX ---

        confidence = np.max(prediction_proba) * 100 # Confidence still based on original probas for display
        prediction_stage = PREDICTION_MAP.get(prediction_numeric, 'Unknown')
        prediction_full_text = FULL_PREDICTION_MAP.get(prediction_stage, 'Unknown Stage')
        current_recommendations = RECOMMENDATIONS.get(prediction_stage, ["No specific recommendations available."])

        # --- BLOCKCHAIN INTEGRATION START ---
        raw_input_data_for_hash = {k: v for k, v in sorted(data.items())}
        input_hash_val = hashlib.sha256(json.dumps(
            raw_input_data_for_hash, sort_keys=True
        ).encode('utf-8')).hexdigest()

        prediction_record = {
            "input_form_hash": input_hash_val,
            "prediction_numeric": int(prediction_numeric),
            "prediction_stage": prediction_stage,
            "confidence_score": float(f"{confidence:.2f}"),
            "timestamp_prediction": str(datetime.now())
        }

        last_block = d_blockchain.get_last_block()
        new_block_index = last_block.index + 1
        new_block = Block(
            new_block_index,
            datetime.now(),
            prediction_record,
            last_block.hash
        )
        if d_blockchain.add_block(new_block):
            print(f"Prediction recorded to blockchain. New block index: {new_block_index}")
            d_blockchain.save_chain_to_file(BLOCKCHAIN_FILE)
        else:
            print(f"Failed to add block {new_block_index} to blockchain. Chain might be invalid.")
        # --- BLOCKCHAIN INTEGRATION END ---

        return render_template('index.html',
                               prediction_result=prediction_stage,
                               prediction_result_full_text=prediction_full_text,
                               confidence_score=f"{confidence:.2f}",
                               recommendations=current_recommendations)

    except Exception as e:
        error_message = f"An error occurred during prediction: {e}. Please check your input values and server logs."
        print(error_message)
        return render_template('index.html', prediction_result="Error", confidence_score="N/A",
                               recommendations=[error_message])

@app.route('/history', methods=['GET'])
def view_history():
    chain_list = d_blockchain.to_list()
    is_valid = d_blockchain.is_chain_valid()
    return render_template('history.html', chain=chain_list, is_valid=is_valid)

@app.route('/blockchain_raw', methods=['GET'])
def get_blockchain_raw():
    chain_list = d_blockchain.to_list()
    is_valid = d_blockchain.is_chain_valid()
    return jsonify({
        'chain': chain_list,
        'length': len(chain_list),
        'is_valid': is_valid
    })

if __name__ == '__main__':
    pass
