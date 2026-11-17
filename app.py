from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
from pathlib import Path
import sklearn 
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(str(BASE_DIR / 'WeightPrediction.pkl'))

TEMPLATES = BASE_DIR / 'templates'

@app.route('/')
def home():
    return send_from_directory(TEMPLATES, 'index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # if request.method == 'POST':
    data = request.get_json()
    height = float(data['height'])
    prediction = model.predict(np.array([[height]]))
    return jsonify({'predicted_weight': round(float(prediction[0]),2)})

if __name__ == '__main__':
    app.run(debug=True)


