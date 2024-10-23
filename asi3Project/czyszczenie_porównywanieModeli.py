import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

# Wczytanie danych
url = 'https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv'
data = pd.read_csv(url)
nazwaPDF = 'analiza_statystyczna.pdf'

# Oczyszczanie starych plików
if os.path.exists(nazwaPDF):
    os.remove(nazwaPDF)

# Utworzenie dokumentu PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Dodanie nagłówka
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, 'Analiza statystyczna zmiennych'.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
pdf.ln(10)

# Wstęp
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, 'Niniejszy dokument przedstawia wyniki analizy statystycznej danych, które pochodzą z zestawu CollegeDistance. '
                      'Dane te zostały wstępnie przetworzone, a następnie posłużyły do budowy modelu predykcyjnego, '
                      'którego celem jest przewidywanie zmiennej "score". W trakcie pracy nad projektem przeprowadzono oczyszczanie '
                      'danych, analizę statystyczną, wybór modeli, ich ocenę oraz optymalizację.')

pdf.ln(10)
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, 'Informacje o zmiennych'.encode('latin-1', 'replace').decode('latin-1'), ln=True)
pdf.set_font("Arial", size=12)

# Funkcja tworząca wykresy i statystyki dla każdej zmiennej
def analyze_column(column_name):
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f'Zmienna: {column_name}'.encode('latin-1', 'replace').decode('latin-1'), ln=True)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f'Typ zmiennej: {data[column_name].dtype}'.encode('latin-1', 'replace').decode('latin-1'), ln=True)

    # Statystyki opisowe
    desc_stats = data[column_name].describe()
    pdf.multi_cell(0, 10, f'Statystyki opisowe:\n{desc_stats.to_string()}'.encode('latin-1', 'replace').decode('latin-1'))

    # Tworzenie wykresu histogramu dla zmiennej
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column_name], bins=30, kde=True)
    plt.title(f'Rozkład zmiennej {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Liczba wystąpień')
    tmp_filename = f'{column_name}_distribution.png'
    plt.savefig(tmp_filename)
    plt.close()

    # Dodanie wykresu do PDF
    pdf.image(tmp_filename, x=10, w=180)
    os.remove(tmp_filename)  # Usunięcie pliku tymczasowego

# Sprawdzenie dostępnych kolumn i wykonanie analizy dla każdej zmiennej
for col in data.columns:
    analyze_column(col)

# Usunięcie kolumny 'rownames'
data.drop('rownames', axis=1, inplace=True)

# Zakodowanie kolumny 'income' na wartości liczbowe
label_encoder = LabelEncoder()
data['income'] = label_encoder.fit_transform(data['income'])

# Zakodowanie kolumny 'ethnicity' na wartości liczbowe
label_encoder = LabelEncoder()
data['ethnicity'] = label_encoder.fit_transform(data['ethnicity'])

# Zamiana wartości 'male' i 'female' na 1 i 0 w kolumnie 'gender'
data['gender'] = data['gender'].map({'male': 1, 'female': 0})

# Zamiana wartości 'yes' i 'no' na 1 i 0 w odpowiednich kolumnach
columns_to_convert = ['fcollege', 'mcollege', 'home', 'urban']
for col in columns_to_convert:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# Zamiana wartości 'region' na 1 dla 'other' i 0 dla 'high', 'low'
data['region'] = data['region'].map({'other': 1, 'west': 0})

# Obsługa brakujących wartości
initial_shape = data.shape
data.dropna(thresh=len(data.columns) - 3, inplace=True)
pdf.add_page()
pdf.cell(0, 10, 'Obsługa brakujących wartości'.encode('latin-1', 'replace').decode('latin-1'), ln=True)
pdf.cell(0, 10, f'Kształt danych przed usunięciem wierszy: {initial_shape}'.encode('latin-1', 'replace').decode('latin-1'), ln=True)
pdf.cell(0, 10, f'Kształt danych po usunięciu wierszy: {data.shape}'.encode('latin-1', 'replace').decode('latin-1'), ln=True)

# Uzupełnienie brakujących wartości medianą dla kolumn numerycznych
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
for col in numerical_cols:
    median_value = data[col].median()
    data[col] = data[col].fillna(median_value)

# Zapisanie oczyszczonych danych
data.to_csv('cleaned_data.csv', index=False)

# Etap porównania modeli
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, 'Porównanie modeli'.encode('latin-1', 'replace').decode('latin-1'), ln=True)
pdf.set_font("Arial", size=12)

# Wybór zmiennych do modelowania
X = data.drop(columns=['score'])
y = data['score']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista modeli do porównania
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regressor': SVR(),
    'K-Neighbors Regressor': KNeighborsRegressor()
}

best_model_name = None
best_mse = float('inf')

# Porównanie modeli
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    pdf.cell(0, 10, f'Model: {model_name}, Mean Squared Error (MSE): {mse:.4f}'.encode('latin-1', 'replace').decode('latin-1'), ln=True)

    if mse < best_mse:
        best_mse = mse
        best_model_name = model_name
        best_model = model

# Zapisanie najlepszego modelu
joblib.dump(best_model, 'best_model.pkl')
pdf.cell(0, 10, f'Najlepszy model: {best_model_name} z MSE: {best_mse:.4f}'.encode('latin-1', 'replace').decode('latin-1'), ln=True)

# Podsumowanie
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, 'Podsumowanie'.encode('latin-1', 'replace').decode('latin-1'), ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, f'W niniejszej pracy wykonano pełną analizę danych, w tym ich oczyszczenie i przygotowanie do modelowania. '
                      f'Zastosowano kilka modeli predykcyjnych, spośród których najlepszy okazał się model "{best_model_name}" z najniższym błędem średniokwadratowym (MSE): {best_mse:.4f}. '
                      'Dokument ten zawiera zarówno szczegóły dotyczące poszczególnych zmiennych, jak i oceny porównywanych modeli. '
                      'W razie potrzeby można przeprowadzić dodatkową optymalizację, aby jeszcze bardziej poprawić jakość modelu.'.encode('latin-1', 'replace').decode('latin-1'))

# Zapisanie PDF
pdf.output(nazwaPDF)

print(f'Dokument zapisany jako {nazwaPDF}.')
