from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the pre-trained model
model = load('logistic_regression_model.joblib')
# Load your dataset with additional details
df = pd.read_csv('/Users/pranjalmishra/Desktop/big_data_analysis/final_dataset.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = pd.DataFrame([data])
        prediction = model.predict(symptoms)[0]
        # Fetch additional details
        details = df[df['prognosis'] == prediction].iloc[0]
        result = {
            'prediction': prediction,
            'Precautions': details['PRECAU'],
            'Risk Factors': details['RISKFAC'],
            'Medicine Name': details['Medicine_Name'],
            'Medicine Composition': details['Medicine_Composition'],
            'Medicine Description': details['Medicine_Description']
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
