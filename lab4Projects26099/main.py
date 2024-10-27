from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Wczytanie modelu
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Endpoint do przewidywania na podstawie danych JSON
@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # Oczekiwanie na dane w formacie JSON
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)  # Konwersja na odpowiedni format
        prediction = model.predict(features)  # Przewidywanie
        return jsonify({'prediction': prediction[0]})

    # Obsługa danych w formacie CSV
    elif 'file' in request.files:
        # Oczekiwanie na plik CSV
        file = request.files['file']
        data = pd.read_csv(file)  # Wczytanie pliku CSV jako DataFrame
        predictions = model.predict(data)  # Przewidywania dla danych w pliku CSV
        return jsonify({'predictions': predictions.tolist()})  # Zwrot przewidywań

    else:
        return jsonify({'error': 'Invalid input format. Please provide JSON or CSV file.'}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
