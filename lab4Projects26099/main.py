from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Wczytanie najlepszego modelu
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Plik z modelem nie został znaleziony.")

# Endpoint do przewidywania
@app.route('/predict', methods=['POST'])
def predict():
    # Sprawdź, czy plik został przesłany
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku w żądaniu'}), 400

    file = request.files['file']

    # Wczytaj plik CSV
    try:
        data = pd.read_csv(file)  # Wczytanie pliku CSV
        features = data.drop(columns=['score'])  # Zakładam, że 'score' to zmienna docelowa
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Przewidywanie
    try:
        predictions = model.predict(features)  # Przewidywanie
        return jsonify({'predictions': predictions.tolist()})  # Zwrot przewidywania
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
