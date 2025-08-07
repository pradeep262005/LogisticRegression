from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['id']),
            float(request.form['age']),
            float(request.form['experience']),
            float(request.form['income']),
            float(request.form['family']),
            float(request.form['ccavg']),
            float(request.form['education']),
            float(request.form['mortgage']),
            float(request.form['cd_account']),
            float(request.form['creditcard'])
        ]

        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)

        result = "Approved ✅" if prediction[0] == 1 else "Not Approved ❌"
        return render_template('index.html', prediction_text=f"Loan Status: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text="Error: Invalid input.")

if __name__ == '__main__':
    app.run(debug=True)
