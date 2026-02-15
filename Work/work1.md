Zadanie 1: Klasyfikacja spamu (MultinomialNB)
To klasyczne zastosowanie algorytmu. Twoim zadaniem jest stworzenie prostego filtra antyspamowego.

Dane: Stwórz mały zbiór danych tekstowych (np. 10 zdań), gdzie każde jest oznaczone jako "spam" lub "ham" (nie-spam).

Cel: 1. Wykorzystaj CountVectorizer z sklearn.feature_extraction.text, aby zamienić tekst na postać liczbową (macierz wystąpień słów).
2. Wytrenuj model MultinomialNB.
3. Przetestuj model na nowym zdaniu, np. "Win a free prize now!".

Kluczowe zagadnienie: Zwróć uwagę, jak model radzi sobie ze słowami, których nie było w zbiorze treningowym.

Zadanie 2: Przewidywanie gatunków irysów (GaussianNB)
W tym zadaniu wykorzystasz dane numeryczne (ciągłe), co wymaga użycia innej wersji algorytmu – Bayesa z rozkładem Gaussa.

Dane: Użyj wbudowanego zbioru danych Iris: from sklearn.datasets import load_iris.

Cel:

Podziel dane na zbiór treningowy i testowy (np. 80/20).

Zastosuj klasyfikator GaussianNB.

Oblicz macierz pomyłek (confusion matrix) oraz dokładność (accuracy) modelu.

Kluczowe zagadnienie: Zastanów się, dlaczego do danych liczbowych (długość i szerokość płatków) używamy GaussianNB, a nie MultinomialNB.

Zadanie 3: Analiza sentymentu recenzji z walidacją krzyżową
Zadanie dla bardziej zaawansowanych, łączące przetwarzanie tekstu z rzetelną oceną modelu.

Dane: Możesz pobrać niewielki zbiór recenzji filmowych lub stworzyć własną listę 20-30 opinii (pozytywnych i negatywnych).

Cel:

Użyj TfidfVectorizer zamiast zwykłego liczenia słów, aby nadać większą wagę istotnym wyrazom.

Zastosuj Pipeline z sklearn.pipeline, aby połączyć wektoryzację z modelem MultinomialNB.

Przeprowadź 5-krotną walidację krzyżową (cross-validation), aby sprawdzić stabilność swojego modelu.

Kluczowe zagadnienie: Sprawdź, jak zmiana parametrów ngram_range w wektoryzatorze (np. analizowanie par słów zamiast pojedynczych wyrazów) wpływa na wynik.

Przydatne biblioteki do importu:
Python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
Zadanie 4: Prognoza pogody i decyzja „Grać czy nie grać?” (BernoulliNB)
To klasyczny przykład z podręczników uczenia maszynowego. Skupia się na cechach binarnych (tak/nie, 0/1).

Dane: Stwórz ramkę danych (np. w pandas), która zawiera kolumny: Słonecznie, Wysoka_Wilgotność, Silny_Wiatr. Wszystkie wartości to 0 lub 1. Etykieta to Zagra w tenisa (0/1).

Cel:

Wykorzystaj BernoulliNB, który jest zoptymalizowany pod kątem cech binarnych.

Przetestuj model dla scenariusza: Jest słonecznie, ale jest silny wiatr i wysoka wilgotność.

Kluczowe zagadnienie: Użycie LabelEncoder z sklearn.preprocessing, jeśli Twoje dane wejściowe to napisy ("Tak"/"Nie"), zamiast liczb.

Zadanie 5: Klasyfikacja wiadomości ze świata (20 Newsgroups)
Czas na „prawdziwe” dane. Biblioteka scikit-learn posiada wbudowane zestawy danych, które są znacznie większe i trudniejsze niż proste przykłady tekstowe.

Dane: Pobierz zbiór newsów: from sklearn.datasets import fetch_20newsgroups. Wybierz np. 4 kategorie (np. sci.space, comp.graphics, rec.sport.baseball, talk.politics.mideast).

Cel:

Zastosuj wektoryzację TfidfVectorizer.

Wytrenuj model MultinomialNB.

Stwórz raport klasyfikacji (classification_report), aby zobaczyć precyzję (precision) i pełność (recall) dla każdej z kategorii osobno.

Kluczowe zagadnienie: Zobacz, które kategorie model myli ze sobą najczęściej (np. czy nauka o kosmosie miesza się z grafiką komputerową?).

Zadanie 6: Problem „Zero Frequency” i parametr Alpha
To zadanie ma na celu zrozumienie, dlaczego w kodzie MultinomialNB(alpha=1.0) parametr alpha jest tak ważny.

Dane: Stwórz bardzo mały zbiór treningowy, w którym słowo "promocja" występuje tylko w spamie, a nigdy w zwykłych wiadomościach.

Cel:

Wytrenuj model z domyślnym alpha=1.0 (wygładzanie Laplace'a).

Spróbuj wytrenować model z alpha=0.0001 (prawie brak wygładzania).

Porównaj prawdopodobieństwa przypisane nowej wiadomości, która zawiera słowo "promocja".

Kluczowe zagadnienie: Zrozumienie, że bez alpha, pojawienie się nowego słowa, którego model nie widział w danej klasie, zeruje całe prawdopodobieństwo (mnożenie przez zero).

Co Ci się przyda:
Narzędzie	Zastosowanie
BernoulliNB	Dane binarne (0/1, występuje/nie występuje).
MultinomialNB	Dane zliczane (częstotliwość słów w tekście).
LabelEncoder	Zamiana etykiet tekstowych na numeryczne.
classification_report	Szczegółowa analityka sukcesu modelu.
Podpowiedź: Jeśli chcesz sprawdzić, jak model radzi sobie „na żywo”, możesz użyć metody predict_proba(). Zamiast gotowej odpowiedzi (Spam/Nie-spam), pokaże Ci ona procentową pewność modelu, np. [0.12, 0.88].

