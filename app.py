# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1 = pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
    return render_template('home.html')

@app.route("/", methods=['POST'])
def predict():
    def safe_float(value):
        try:
            return float(value)
        except ValueError:
            return 0.0

    def safe_str(value):
        return value if value else 'unknown'

    # All input fields
    input_tenure = safe_float(request.form.get('tenure', '0'))
    input_monthly_charges = safe_float(request.form.get('monthly_charges', '0'))
    input_internet_service = safe_str(request.form.get('internet_service', 'unknown'))
    input_online_security = safe_str(request.form.get('online_security', 'unknown'))
    input_online_backup = safe_str(request.form.get('online_backup', 'unknown'))
    input_device_protection = safe_str(request.form.get('device_protection', 'unknown'))
    input_tech_support = safe_str(request.form.get('tech_support', 'unknown'))
    input_streaming_tv = safe_str(request.form.get('streaming_tv', 'unknown'))
    input_contract_type = safe_str(request.form.get('contract_type', 'unknown'))

    data = {
        'tenure': input_tenure,
        'MonthlyCharges': input_monthly_charges,
        'InternetService': input_internet_service,
        'OnlineSecurity': input_online_security,
        'OnlineBackup': input_online_backup,
        'DeviceProtection': input_device_protection,
        'TechSupport': input_tech_support,
        'StreamingTV': input_streaming_tv,
        'ContractType': input_contract_type
    }

    df = pd.DataFrame([data])
    df_dummies = pd.get_dummies(df)

    # Load model and make prediction
    model = pickle.load(open("model.sav", "rb"))
    prediction = model.predict(df_dummies)
    probability = model.predict_proba(df_dummies)[:, 1]

    probability_value = probability[0] if len(probability) > 0 else 0

    if prediction == 1:
        output1 = "This customer is likely to be churned!!"
    else:
        output1 = "This customer is likely to continue!!"
    
    output2 = "Confidence: {:.2f}%".format(probability_value * 100)

    return render_template('home.html', output1=output1, output2=output2,
                           tenure=input_tenure,
                           monthly_charges=input_monthly_charges,
                           internet_service=input_internet_service,
                           online_security=input_online_security,
                           online_backup=input_online_backup,
                           device_protection=input_device_protection,
                           tech_support=input_tech_support,
                           streaming_tv=input_streaming_tv,
                           contract_type=input_contract_type)

if __name__ == "__main__":
    app.run(debug=True)
