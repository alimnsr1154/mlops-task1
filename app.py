# app.py

from flask import Flask, request, jsonify
from model import train_model

app = Flask(__name__)
model = train_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

