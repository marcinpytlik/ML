import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Titanic AI Predictor", page_icon="🚢")

# --- FUNKCJE POMOCNICZE ---
@st.cache_resource # Zapamiętuje model, żeby nie wczytywać go przy każdym kliknięciu
def load_or_train_model():
    model_path = 'model_titanica.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Fallback: szybki trening jeśli brak pliku
        df = sns.load_dataset('titanic')
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
        X = df[features].copy()
        X['sex'] = X['sex'].map({'male': 0, 'female': 1})
        X['age'] = X['age'].fillna(X['age'].mean())
        y = df['survived']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        return model

model = load_or_train_model()

# --- INTERFEJS UŻYTKOWNIKA ---
st.title("🚢 Czy przeżyłbyś katastrofę Titanica?")
st.write("Wprowadź swoje dane poniżej, a nasz algorytm Lasu Losowego obliczy Twoje szanse.")

# Tworzymy dwie kolumny dla lepszego wyglądu
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Klasa biletu", [1, 2, 3], help="1 - Najwyższa, 3 - Ekonomiczna")
    sex = st.radio("Płeć", ["Mężczyzna", "Kobieta"])
    age = st.slider("Wiek", 0, 100, 25)

with col2:
    sibsp = st.number_input("Liczba rodzeństwa/małżonków", 0, 10, 0)
    parch = st.number_input("Liczba rodziców/dzieci", 0, 10, 0)
    fare = st.number_input("Cena biletu (w ówczesnych funtach)", 0.0, 600.0, 32.0)

# --- PREDYKCJA ---
st.divider()

if st.button("Sprawdź mój los", type="primary"):
    # Przygotowanie danych do modelu
    sex_val = 1 if sex == "Kobieta" else 0
    input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare]], 
                              columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'])
    
    # Obliczenia
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Wyświetlanie wyników
    if prediction == 1:
        st.success(f"### GRATULACJE! Przeżyjesz!")
        st.balloons()
    else:
        st.error(f"### NIESTETY... Prawdopodobnie zginiesz.")
    
    st.write(f"Twoja szansa na ratunek wynosi: **{probability * 100:.1f}%**")

    # Dodatkowa informacja dla użytkownika
    st.info("Pamiętaj, że to tylko model statystyczny oparty na historycznych danych.")