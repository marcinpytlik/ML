import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. ŁADOWANIE DANYCH
    print("Pobieranie danych o pasażerach Titanica...")
    df = sns.load_dataset('titanic')

    # 2. PREPROCESSING (Przygotowanie danych)
    # Wybieramy kluczowe cechy
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    X = df[features].copy()
    y = df['survived']

    # Mapowanie płci na liczby (0 = mężczyzna, 1 = kobieta)
    X['sex'] = X['sex'].map({'male': 0, 'female': 1})

    # Uzupełnianie brakujących danych o wieku średnią wartością
    X['age'] = X['age'].fillna(X['age'].mean())

    # 3. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. TRENOWANIE MODELU (Las Losowy)
    print("Trenowanie modelu Lasu Losowego (100 drzew)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. EWALUACJA MODELU
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\n--- WYNIK MODELU ---")
    print(f"Skuteczność (Accuracy): {acc * 100:.2f}%")

    # 6. WIZUALIZACJA: CO BYŁO NAJWAŻNIEJSZE?
    pokaz_waznosc_cech(model, features)

    # 7. INTERAKTYWNA PREDYKCJA DLA UŻYTKOWNIKA
    interaktywny_test(model)

def pokaz_waznosc_cech(model, feature_names):
    """Generuje wykres słupkowy pokazujący, które dane miały największy wpływ na wynik."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title('Ważność cech w modelu Lasu Losowego (Titanic)')
    plt.xlabel('Wpływ na decyzję modelu')
    plt.ylabel('Cechy')
    print("\n[Wyświetlam wykres ważności cech...]")
    plt.show()

def interaktywny_test(model):
    """Pozwala użytkownikowi wprowadzić własne dane do modelu."""
    print("\n--- TEST TWOJEGO PRZETRWANIA ---")
    try:
        pclass = int(input("Klasa biletu (1-najlepsza, 2, 3-ekonomiczna): "))
        plec_str = input("Płeć (m/k): ").lower()
        age = float(input("Wiek: "))
        sibsp = int(input("Liczba rodzeństwa/małżonków na pokładzie: "))
        parch = int(input("Liczba rodziców/dzieci na pokładzie: "))
        fare = float(input("Cena biletu (np. 10-500): "))

        sex = 1 if plec_str == 'k' else 0

        # Tworzenie ramki danych dla predykcji
        user_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], 
                                 columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'])

        # Predykcja szansy
        prob = model.predict_proba(user_data)[0][1]
        decision = model.predict(user_data)[0]

        print("\n" + "="*40)
        if decision == 1:
            print(f"WERDYKT: PRZEŻYŁEŚ/AŚ!")
        else:
            print(f"WERDYKT: NIESTETY, ZGINĄŁEŚ/AŚ.")
        
        print(f"Szansa na ratunek wg modelu: {prob * 100:.2f}%")
        print("="*40)

    except Exception as e:
        print(f"Wystąpił błąd w danych: {e}")

if __name__ == "__main__":
    main()