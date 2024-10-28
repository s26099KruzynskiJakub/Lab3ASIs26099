# Lab4 Analizator Wyników

## Spis treści
1. [Opis projektu](#opis-projektu)
2. [Jak sklonować repozytorium](#jak-sklonować-repozytorium)
3. [Jak uruchomić aplikację lokalnie](#jak-uruchomić-aplikację-lokalnie)
4. [Jak uruchomić aplikację z wykorzystaniem Dockera](#jak-uruchomić-aplikację-z-wykorzystaniem-dockera)
5. [Korzystanie z obrazu Docker z Docker Hub](#korzystanie-z-obrazu-docker-z-docker-hub)

## Opis projektu
Aplikacja analizuje wyniki na podstawie dostarczonych danych. Umożliwia użytkownikowi przesyłanie plików CSV i otrzymywanie prognoz.

## Jak sklonować repozytorium
Aby sklonować to repozytorium, wykonaj następujące kroki:

1. Otwórz terminal.
2. Użyj polecenia `git clone`, aby sklonować repozytorium:
   ```bash
   git clone https://github.com/s26099KruzynskiJakub/Lab4ASIs26099.git
Przejdź do katalogu projektu:
bash

cd lab4_analizator_wynikow
Jak uruchomić aplikację lokalnie
Aby uruchomić aplikację lokalnie, wykonaj poniższe kroki:

Upewnij się, że masz zainstalowany Python oraz pip.
Zainstaluj zależności:
bash

pip install -r requirements.txt
Uruchom aplikację:
bash

python main.py
Aplikacja powinna być dostępna pod adresem http://localhost:5000.
Jak uruchomić aplikację z wykorzystaniem Dockera
Aby uruchomić aplikację z użyciem Dockera, wykonaj następujące kroki:

Upewnij się, że masz zainstalowanego Dockera.
Zbuduj obraz Dockera:
bash

docker build -t kubeczeg0/lab4_analizator_wynikow:latest .
Uruchom kontener:
bash

docker run -d -p 5000:5000 kubeczeg0/lab4_analizator_wynikow:latest
Aplikacja powinna być dostępna pod adresem http://localhost:5000.
Korzystanie z obrazu Docker z Docker Hub
Aby korzystać z obrazu Dockera, który znajduje się na Docker Hub, wykonaj następujące kroki:

Pobierz obraz z Docker Hub:
bash

docker pull kubeczeg0/lab4_analizator_wynikow:latest
Uruchom kontener z pobranego obrazu:
bash

docker run -d -p 5000:5000 kubeczeg0/lab4_analizator_wynikow:latest
Sprawdź działanie aplikacji: Możesz przetestować aplikację, wysyłając zapytanie do endpointu:
bash
(jeżeli przestrzegane byly poprzednie instrukcje osoba powinna
 znaleźć się w katalogu projektu w którym jest plik csv z danymi dla modelu)

curl -X POST -F "file=@cleaned_data.csv" http://localhost:5000/predict
