import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from collections import Counter

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Bayesowski Multitool", page_icon="🧠", layout="wide")

# --- LOGIKA AUTOKOREKTY (NAIVE BAYES) ---
WORDS = Counter(['który', 'którzy', 'kuter', 'kutry', 'matematyka', 'python', 'programowanie', 'algorytm', 'dane', 'szkoła'])

def P(word, N=sum(WORDS.values())): return WORDS[word] / N

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyząćęłńóśźż'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def candidates(word): 
    return (set([word]) if word in WORDS else None) or set(w for w in edits1(word) if w in WORDS) or [word]

# --- MENU BOCZNE ---
st.sidebar.title("🧠 Menu Algorytmów")
choice = st.sidebar.selectbox("Wybierz zastosowanie:", 
    ["Autokorekta", "Predictive Maintenance", "Real-Time Bidding", "System Rekomendacji"])

st.sidebar.divider()
st.sidebar.write("**Wzór Bayesa:**")
st.sidebar.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}")

# --- 1. AUTOKOREKTA ---
if choice == "Autokorekta":
    st.title("🔤 Inteligentna Autokorekta")
    st.write("Algorytm szuka słowa o najwyższym prawdopodobieństwie $P(Słowo|Błąd)$.")
    
    text_input = st.text_input("Wpisz słowo (spróbuj 'ktury' lub 'pyton'):").lower()
    
    if text_input:
        prediction = max(candidates(text_input), key=P)
        if prediction != text_input:
            st.success(f"Czy chodziło Ci o: **{prediction}**?")
        else:
            st.info("Słowo wydaje się poprawne lub nie ma go w słowniku.")

# --- 2. PREDICTIVE MAINTENANCE ---
elif choice == "Predictive Maintenance":
    st.title("🏭 Wykrywanie Awarii Maszyn")
    st.write("Wykorzystujemy **Gaussian Naive Bayes** do analizy szumu z czujników.")

    # Trening "na żywo"
    X_train = np.array([[60, 2], [65, 3], [70, 2], [110, 8], [115, 9], [120, 10], [68, 4]])
    y_train = np.array([0, 0, 0, 1, 1, 1, 0])
    model = GaussianNB().fit(X_train, y_train)

    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperatura turbiny (°C)", 50, 150, 70)
    with col2:
        vib = st.slider("Poziom wibracji", 0, 15, 3)

    prob = model.predict_proba([[temp, vib]])[0][1]
    
    st.metric("Ryzyko awarii", f"{prob*100:.1f}%")
    if prob > 0.5:
        st.error("⚠️ WYKRYTO ZAGROŻENIE! Wymagany przegląd.")
    else:
        st.success("✅ Maszyna pracuje w normie.")

# --- 3. REAL-TIME BIDDING ---
elif choice == "Real-Time Bidding":
    st.title("💰 Aukcja Reklamowa (RTB)")
    st.write("Szybka decyzja: licytować wyświetlenie reklamy temu użytkownikowi?")

    # Dane: [Zalogowany, Sklep_Wczoraj, Mobile]
    X_ads = [[1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1]]
    y_ads = [1, 0, 0, 1, 0]
    model_ads = BernoulliNB().fit(X_ads, y_ads)

    st.subheader("Profil użytkownika:")
    is_logged = st.checkbox("Zalogowany")
    visited = st.checkbox("Odwiedził sklep wczoraj")
    is_mobile = st.checkbox("Urządzenie mobilne")

    if st.button("Oblicz opłacalność aukcji"):
        features = [[int(is_logged), int(visited), int(is_mobile)]]
        p_click = model_ads.predict_proba(features)[0][1]
        
        st.write(f"Prawdopodobieństwo kliknięcia: **{p_click*100:.1f}%**")
        if p_click > 0.4:
            st.success("LICYTUJ! To wartościowy profil.")
        else:
            st.warning("ODPUŚĆ. Mała szansa na konwersję.")

# --- 4. SYSTEM REKOMENDACJI ---
elif choice == "System Rekomendacji":
    st.title("🎬 Rekomendacje Filmowe")
    st.write("Prawdopodobieństwo warunkowe: Co polecić widzowi?")

    filmy = {
        "Batman": {"widzowie": 500, "oba": 350},
        "Gwiezdne Wojny": {"widzowie": 600, "oba": 120},
        "Incepcja": {"widzowie": 450, "oba": 300}
    }
    
    wybrany = st.selectbox("Film, który Ci się podobał:", list(filmy.keys()))
    
    p_rekomendacji = filmy[wybrany]["oba"] / filmy[wybrany]["widzowie"]
    
    st.write(f"Na podstawie Twojego wyboru, szansa że polubisz **Jokera** wynosi:")
    st.progress(p_rekomendacji)
    st.write(f"**{p_rekomendacji*100:.0f}%**")