import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- 1. PRZYGOTOWANIE DANYCH (Mały zbiór treningowy) ---
data = {
    'text': [
        'Hey, are we still meeting for coffee today?',
        'Your invoice for 500 USD is attached. Please pay now.',
        'Claim your free prize! Click this link to win 1000 dollars!',
        'Can you send me the report by 5 PM?',
        'CONGRATULATIONS! You have won a free iPhone. Call now!',
        'Meeting rescheduled to Monday morning.',
        'Urgent: Your account has been hacked. Verify your password.',
        'Just checking in to see how you are doing.',
        'Double your income working from home! No experience needed.'
    ],
    'label': ['ham', 'spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

# --- 2. BUDOWA PIPELINE'U ---
# Pipeline łączy dwa kroki: zamianę tekstu na liczby i algorytm Bayesa
@st.cache_resource
def train_spam_model():
    model = Pipeline([
        ('vectorizer', CountVectorizer()), # Zamiana tekstu na wektory liczb
        ('nb', MultinomialNB())            # Wielomianowy Naiwny Bayes
    ])
    model.fit(df['text'], df['label'])
    return model

model = train_spam_model()

# --- 3. INTERFEJS STREAMLIT ---
st.set_page_config(page_title="Anty-Spam AI", page_icon="📧")

st.title("📧 Inteligentny Filtr Spamu")
st.write("Wpisz treść wiadomości, a Algorytm Bayesa sprawdzi, czy to bezpieczny mail.")

# Pole tekstowe dla użytkownika
user_input = st.text_area("Treść wiadomości:", placeholder="Np. Win a free prize now!")

if st.button("Analizuj wiadomość"):
    if user_input.strip() == "":
        st.warning("Wpisz jakąś wiadomość!")
    else:
        # Predykcja
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]
        
        # Wyświetlanie wyników
        st.divider()
        if prediction == 'spam':
            st.error(f"### 🚩 TO JEST SPAM!")
            st.write(f"Prawdopodobieństwo spamu: **{proba[1]*100:.2f}%**")
        else:
            st.success(f"### ✅ TO JEST BEZPIECZNA WIADOMOŚĆ")
            st.write(f"Prawdopodobieństwo, że to zwykły mail: **{proba[0]*100:.2f}%**")

# Sekcja edukacyjna
with st.expander("Jak to działa?"):
    st.write("""
    Model oblicza prawdopodobieństwo wystąpienia słów w obu kategoriach.
    Jeśli słowa takie jak **'free'**, **'win'** lub **'urgent'** pojawiają się częściej, 
    wynik równania Bayesa przesuwa się w stronę spamu:
    """)
    st.latex(r'''P(Spam|Słowo) = \frac{P(Słowo|Spam) \cdot P(Spam)}{P(Słowo)}''')