from flask import Flask, request, jsonify
import pickle
import numpy as np

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Wczytanie modelu
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Endpoint do przewidywania
@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type == 'application/json':
        # Obsługa JSON
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
    elif request.content_type == 'text/csv':
        # Obsługa CSV
        file = request.files['file']
        features = np.loadtxt(file, delimiter=",").reshape(1, -1)
    else:
        return jsonify({'error': 'Unsupported format. Use JSON or CSV.'}), 400

    # Przewidywanie
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
