from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize Flask app
app = Flask(__name__)

# Define dataset path
DATASET_PATH = r'C:\\Users\\Dell\\Desktop\\Frud_detection\\onlinefraud.csv'  # Ensure this file exists

# Load dataset safely
df = None  # Initialize as None

if os.path.exists(DATASET_PATH):  # Check if file exists
    try:
        df = pd.read_csv(DATASET_PATH)
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print("❌ Error loading dataset:", str(e))
else:
    print(f"❌ ERROR: Dataset file not found at {DATASET_PATH}")

# Load trained models
try:
    xgb_model = pickle.load(open('./models/xgb.sav', 'rb'))
    lr_model = pickle.load(open('./models/lr.sav', 'rb'))
    print("✅ Models loaded successfully!")
except Exception as e:
    print("❌ Error loading models:", str(e))

# Define important features
important_features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                      'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 
                      'nameOrig', 'nameDest']

# Preprocessing function
def preprocess_input(data):
    try:
        categorical_features = ['type', 'nameOrig', 'nameDest']  # Encode these
        numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                            'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']

        # Transform categorical and numeric features
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        processed_data = pipeline.fit_transform(pd.DataFrame([data], columns=important_features))
        return processed_data
    except Exception as e:
        return str(e)

# Home route
@app.route('/')
def home():
    return render_template('index.html', features=important_features)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        input_data = {feature: request.form[feature] for feature in important_features}

        # Convert numeric values
        numeric_keys = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
        for key in numeric_keys:
            input_data[key] = float(input_data[key])

        # Debugging logs
        print("🟢 Input Data:", input_data)

        # Preprocess input
        final_features = preprocess_input(input_data)
        print("🟢 Processed Features Shape:", final_features.shape)

        # Make predictions
        fraud_prediction = xgb_model.predict(final_features)[0]
        fraud_prob = xgb_model.predict_proba(final_features)[0][1]

        return render_template('index.html', 
                               prediction_text=f'Fraud Prediction: {"Fraud" if fraud_prediction else "Not Fraud"}',
                               probability_text=f'Fraud Probability: {fraud_prob:.2f}',
                               features=important_features)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)