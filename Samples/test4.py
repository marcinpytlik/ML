import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# --- KONFIGURACJA ---
MODEL_FILE = 'model_bayes_titanic.pkl'

@st.cache_resource
def get_model():
    # Sprawdzamy czy model istnieje, jeśli nie - trenujemy go
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        # Pobranie i przygotowanie danych
        df = sns.load_dataset('titanic')
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
        X = df[features].copy()
        X['sex'] = X['sex'].map({'male': 0, 'female': 1})
        X['age'] = X['age'].fillna(X['age'].mean())
        y = df['survived']
        
        # Trening klasyfikatora Bayesa
        model = GaussianNB()
        model.fit(X, y)
        
        joblib.dump(model, MODEL_FILE)
        return model

# --- INTERFEJS STRONY ---
st.set_page_config(page_title="Bayes Titanic Predictor", page_icon="⚖️")

st.title("⚖️ Klasyfikator Bayesa: Przetrwanie na Titanicu")
st.markdown("""
Ten model używa **prawdopodobieństwa warunkowego**, aby ocenić Twoje szanse. 
W przeciwieństwie do Lasów Losowych, Bayes traktuje każdą cechę (wiek, płeć) jako niezależny czynnik.
""")

# Boczne menu z informacjami o algorytmie
st.sidebar.header("O Algorytmie")
st.sidebar.write("""
**Naiwny Bayes** wylicza prawdopodobieństwo na podstawie wzoru:
""")
st.sidebar.latex(r'''P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}''')

# --- FORMULARZ DANYCH ---
with st.container():
    st.subheader("Wprowadź swoje dane:")
    c1, c2 = st.columns(2)
    
    with c1:
        pclass = st.selectbox("Klasa podróży", [1, 2, 3])
        sex = st.radio("Płeć", ["Mężczyzna", "Kobieta"])
        age = st.slider("Twój wiek", 1, 100, 25)
    
    with c2:
        sibsp = st.number_input("Rodzeństwo/Małżonkowie", 0, 10, 0)
        parch = st.number_input("Rodzice/Dzieci", 0, 10, 0)
        fare = st.number_input("Opłata za bilet", 0.0, 512.0, 32.0)

# --- OBLICZENIA ---
model = get_model()

if st.button("Oblicz prawdopodobieństwo metodą Bayesa", type="primary"):
    # Przygotowanie danych wejściowych
    sex_val = 1 if sex == "Kobieta" else 0
    input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare]], 
                              columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'])
    
    # Predykcja
    prob = model.predict_proba(input_data)[0] # [szansa na śmierć, szansa na przeżycie]
    survival_chance = prob[1] * 100
    
    # Wyświetlenie wyniku
    st.divider()
    st.metric(label="Szansa na przeżycie", value=f"{survival_chance:.2f}%")
    
    if survival_chance > 50:
        st.success("Model sugeruje, że statystycznie masz wysokie szanse na ratunek!")
    else:
        st.error("Model sugeruje, że statystycznie Twoje szanse były niskie.")

    # Wykres prawdopodobieństwa
    chart_data = pd.DataFrame({
        'Los': ['Zginąłbyś', 'Przeżyłbyś'],
        'Prawdopodobieństwo': [prob[0], prob[1]]
    })
    st.bar_chart(data=chart_data, x='Los', y='Prawdopodobieństwo')