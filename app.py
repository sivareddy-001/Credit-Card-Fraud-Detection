from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open('rf_model.pkl', 'rb') as f:
    rf_loaded = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)



# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        transaction_amount = float(request.form['transaction_amount'])
        transaction_time = float(request.form['transaction_time'])
        customer_age = int(request.form['customer_age'])
        merchant_category = int(request.form['merchant_category'])
        foreign_transaction = int(request.form['foreign_transaction'])
        location_mismatch = int(request.form['location_mismatch'])
        device_trust_score = float(request.form['device_trust_score'])
        velocity_last_24h = int(request.form['velocity_last_24h'])

        # Create input dictionary with all features = 0
        #input_data = {col: 0 for col in feature_columns}

        features=[[transaction_amount,merchant_category ,transaction_time, customer_age, foreign_transaction, location_mismatch, device_trust_score, velocity_last_24h]]

        
        # # One-hot encode merchant_category
        # category_col = f"merchant_category_{merchant_category}"
        # if category_col in input_data:
        #     input_data[category_col] = 1
        features=np.array(features)
       
        features=scaler.transform(features)  # Scale features if you used scaling


        # Predict using Random Forest
        rf_pred = rf_loaded.predict(features)[0]
        rf_prob = rf_loaded.predict_proba(features)[0][1]

        
        # # Predict using Logistic Regression
        # lr_pred = lr_loaded.predict(features)[0]
        # lr_prob = lr_loaded.predict_proba(features)[0][1]

        
        # Convert prediction to label
        rf_result = "Fraudulent Transaction" if rf_pred == 1 else "Legitimate Transaction"
        #lr_result = "Fraudulent Transaction" if lr_pred == 1 else "Legitimate Transaction"

        return render_template(
            'index.html',
            rf_prediction=rf_result,
            rf_probability=rf_prob * 100
            #lr_prediction=lr_result,
            #lr_probability=round(lr_prob * 100, 2)
        )

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


# Run App
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)