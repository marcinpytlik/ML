import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Nazwa pliku, w którym zapiszemy nasz "mózg" AI
MODEL_FILE = 'model_titanica.pkl'

def przygotuj_dane():
    """Pobiera i czyści dane wejściowe."""
    df = sns.load_dataset('titanic')
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    
    X = df[features].copy()
    y = df['survived']

    # Konwersja danych tekstowych na liczbowe
    X['sex'] = X['sex'].map({'male': 0, 'female': 1})
    # Uzupełnienie braków w wieku średnią
    X['age'] = X['age'].fillna(X['age'].mean())
    
    return X, y, features

def trenuj_i_zapisz_model(X, y):
    """Trenuje model i zapisuje go na dysku."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Rozpoczynam trenowanie modelu...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Ewaluacja
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model wytrenowany! Skuteczność: {acc * 100:.2f}%")
    
    # ZAPIS MODELU
    joblib.dump(model, MODEL_FILE)
    print(f"Model został zapisany w pliku: {MODEL_FILE}")
    return model

def pokaz_waznosc_cech(model, feature_names):
    """Tworzy wykres pokazujący, co było kluczowe dla modelu."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='magma')
    plt.title('Kluczowe czynniki przetrwania według AI')
    plt.show()

def interaktywna_predykcja(model):
    """Pozwala użytkownikowi na testowanie modelu w konsoli."""
    print("\n--- SPRAWDŹ SWOJE SZANSE NA TITANICU ---")
    try:
        pclass = int(input("Klasa (1-3): "))
        plec = 1 if input("Płeć (m/k): ").lower() == 'k' else 0
        age = float(input("Wiek: "))
        rodzina = int(input("Liczba członków rodziny na pokładzie: "))
        bilet = float(input("Cena biletu (średnio 32): "))

        dane_osoby = pd.DataFrame([[pclass, plec, age, rodzina, 0, bilet]], 
                                 columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'])

        szansa = model.predict_proba(dane_osoby)[0][1]
        wynik = "PRZEŻYJESZ" if szansa > 0.5 else "ZGINIESZ"
        
        print(f"\nWynik: {wynik} (Prawdopodobieństwo ratunku: {szansa*100:.1f}%)")
    except Exception as e:
        print(f"Błąd danych: {e}")

def main():
    X, y, cechy = przygotuj_dane()

    # Sprawdzamy, czy mamy już zapisany model
    if os.path.exists(MODEL_FILE):
        print(f"Znaleziono zapisany model. Wczytywanie z {MODEL_FILE}...")
        model = joblib.load(MODEL_FILE)
    else:
        print("Brak zapisanego modelu.")
        model = trenuj_i_zapisz_model(X, y)

    # Wizualizacja
    pokaz_waznosc_cech(model, cechy)
    
    # Testowanie
    interaktywna_predykcja(model)

if __name__ == "__main__":
    main()