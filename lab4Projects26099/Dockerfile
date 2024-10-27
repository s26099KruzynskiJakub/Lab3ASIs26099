# Wybór podstawowego obrazu
FROM python:3.9-slim

# Ustawienie katalogu roboczego
WORKDIR /app

# Skopiowanie plików aplikacji
COPY requirements.txt .
COPY main.py .
COPY best_model.pkl .

# Instalacja zależności
RUN pip install -r requirements.txt

# Określenie portu
EXPOSE 5000

# Uruchomienie aplikacji Flask
CMD ["python", "main.py"]
