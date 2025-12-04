"""
Projekt: Klasyfikacja danych – Decision Tree i SVM
Autorzy: Marek Lewańczyk s29420, Katarzyna Kasperek s27553
Opis:
  Skrypt realizuje kompletny pipeline uczenia maszynowego dla dowolnego
  zbioru danych: wczytanie, czyszczenie, preprocessing, PCA, trening
  klasyfikatorów (DecisionTree, SVM), zapis wykresów, generowanie metryk.

Instrukcja użycia:
  1. Umieść plik CSV/TXT w katalogu /data lub podaj URL.
  2. Ustaw wartość FILE_PATH na ścieżkę do zbioru.
  3. Uruchom: python main.py
  4. Wygenerowane wykresy znajdziesz w katalogu roboczym.

Referencje:
  - Scikit‑learn API: https://scikit-learn.org/
  - Matplotlib: https://matplotlib.org/
  - Seaborn: https://seaborn.pydata.org/
"""

import pandas as pd  # import biblioteki pandas do pracy z tabelami (DataFrame)
import matplotlib.pyplot as plt  # import modułu do rysowania wykresów (matplotlib)
import seaborn as sns  # import biblioteki seaborn do ładniejszych wykresów
import warnings  # import modułu do obsługi ostrzeżeń (wyciszanie itp.)

from sklearn.model_selection import train_test_split  # import funkcji dzielącej dane na trening/test
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # import scalerów i enkoderów cech
from sklearn.compose import ColumnTransformer  # import narzędzia do jednoczesnego przetwarzania kolumn
from sklearn.pipeline import Pipeline  # import pomocnika do łączenia kroków w pipeline
from sklearn.decomposition import PCA  # import PCA do redukcji wymiarów
from sklearn.tree import DecisionTreeClassifier  # import klasyfikatora drzewa decyzyjnego
from sklearn.svm import SVC  # import klasyfikatora SVM
from sklearn.metrics import (  # import metryk do oceny modeli
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")  # wyłącz wszystkie ostrzeżenia, żeby wyjście było czytelniejsze

"""
Ten moduł implementuje kompletny pipeline ML dla klasyfikacji.
Funkcjonalności:
  - automatyczne wykrywanie kolumny celu,
  - czyszczenie i synchronizacja danych,
  - preprocessing (Standaryzacja + OneHotEncoding),
  - wizualizacja PCA 2D,
  - trenowanie drzewa decyzyjnego,
  - trenowanie SVM z różnymi kernelami,
  - zapis wykresów do PNG,
  - generowanie metryk i przykładowych predykcji.

Kod zaprojektowany tak, aby działał dla dowolnego zbioru danych bez
modyfikacji logiki – wystarczy podmienić FILE_PATH.
"""


def main():  # główna funkcja uruchamiająca cały pipeline
    """
    Główna funkcja uruchamiająca pipeline ML.
    Wczytuje dane, wykonuje preprocessing, PCA, trenuje modele
    oraz generuje raporty tekstowe i graficzne.
    """
    FILE_PATH = "data/titanic.csv"  # ta zmienna przechowuje ścieżkę lub URL do pliku z danymi

    try:
        df = pd.read_csv(FILE_PATH, sep=None, engine="python")      # spróbuj wczytać plik i dopasować separator automatycznie
    except:
        df = pd.read_csv(FILE_PATH)     # jeśli powyższe się nie uda, wczytaj standardowo bez auto-separatora

    # Przejdź przez wszystkie kolumny i zamień przecinki na kropki w wartościach tekstowych
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)    # zastąp przecinki kropkami, przydatne przy liczbach w formacie europejskim

    # Spróbuj przekonwertować każdą kolumnę na typ float, jeśli to możliwe
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)            # konwertuj kolumnę do float, jeżeli zawiera liczby
        except:
            pass            # jeżeli konwersja się nie powiedzie (np. tekst), zostaw kolumnę bez zmian

    print("\nZaładowano dane")    # komunikat informujący, że dane zostały wczytane
    print(df.head())    # pokaż pierwsze pięć wierszy tabeli, żeby zobaczyć przykładowe dane
    print("\nKształt danych:", df.shape)    # pokaż rozmiar danych (liczba wierszy, liczba kolumn)

    # ============================================================
    # 2. Automatyczna identyfikacja kolumny celu
    # ============================================================

    possible_targets = ["target", "label", "class", "y", "Y"]  # lista możliwych nazw kolumny celu
    target_col = None  # zmienna, w której przechowamy nazwę kolumny celu
    for col in df.columns:
        if col.lower() in possible_targets:
            target_col = col
            break  # jeżeli znajdziemy pasującą nazwę, przerwij pętlę

    if target_col is None:
        target_col = df.columns[-1]  # jeśli nie znaleziono typowej nazwy, użyj ostatniej kolumny jako celu

    print("\nKolumna celu:", target_col)  # pokaż, którą kolumnę program uznał za cel

    y = df[target_col]  # wektor etykiet/targetów
    X = df.drop(columns=[target_col])  # macierz cech (wszystko poza kolumną celu)

    # ============================================================
    # 3. Konwersja regresji → klasyfikacja, jeśli potrzeba
    # ============================================================

    if (y.dtype.kind in 'bifc') and (y.nunique() > 10):
        print("\nTarget jest numeryczny → konwersja na klasy binarne (threshold = mediana).")  # informacja co robimy
        threshold = y.median()  # oblicz medianę etykiet
        y = (y > threshold).astype(int)  # zamień wartości na 0/1 względem mediany

    print("Liczba klas:", y.nunique())  # wypisz liczbę unikalnych klas w target

    # ============================================================
    # 4. Czyszczenie danych + synchronizacja X i y
    # ============================================================

    mask = X.dropna().index  # zachowaj tylko indeksy wierszy bez brakujących wartości
    X = X.loc[mask].reset_index(drop=True)  # zastosuj maskę do X i zresetuj indeksy
    y = y.loc[mask].reset_index(drop=True)  # zastosuj maskę do y i zresetuj indeksy

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()  # lista kolumn numerycznych
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()  # lista kolumn kategorycznych (tekstowych)

    print("\nKolumny numeryczne:", numeric_cols)  # pokaż, które kolumny są liczbowe
    print("Kolumny kategoryczne:", categorical_cols)  # pokaż, które kolumny są tekstowe/kategoryczne

    # ============================================================
    # 5. Preprocessing
    # ============================================================

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),  # dla kolumn numerycznych: standaryzuj (0 średnia, 1 odchylenie)
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)  # dla kategorycznych: one-hot encoding, ignoruj nieznane kategorie
    ])

    # ============================================================
    # 6. Wizualizacje + zapisywanie do PNG
    # ============================================================

    # Scatterplot or histogram / simple overview
    plt.figure(figsize=(8,6))  # utwórz nową figurę o rozmiarze 8x6 cala
    if "X" in X.columns and "Y" in df.columns:
        # tylko jeśli kolumny 'X' i 'Y' istnieją w danych, narysuj scatter plot
        sns.scatterplot(data=df, x="X", y="Y", hue=y, palette="coolwarm", legend='full')  # scatter plot kolorowany według targetu
        plt.title("Scatter plot X vs target / Y")  # dodaj tytuł wykresu
        plt.savefig("scatter_plot.png", bbox_inches="tight", dpi=150)  # zapisz wykres do pliku PNG
        plt.close()  # zamknij figurę, żeby zwolnić pamięć
    else:
        # w przeciwnym razie narysuj histogram rozkładu klas
        sns.histplot(y, bins=y.nunique())  # histogram pokazujący liczbę przykładów w każdej klasie
        plt.title("Rozkład klas (y)")  # tytuł wykresu
        plt.savefig("histogram_labels.png", bbox_inches="tight", dpi=150)  # zapisz histogram do pliku PNG
        plt.close()  # zamknij figurę

    # PCA 2D jeśli więcej niż 1 cecha
    if X.shape[1] > 1:
        X_proc = preprocess.fit_transform(X)  # dopasuj preprocess i przetwórz X
        pca = PCA(n_components=2)  # utwórz PCA redukujące do 2 składowych
        X_pca = pca.fit_transform(X_proc)  # dopasuj PCA i przekształć dane

        plt.figure(figsize=(8,6))  # nowa figura do wykresu PCA
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="coolwarm", legend='full')  # wykres punktowy PC1 vs PC2
        plt.title("PCA 2D – wizualizacja danych")  # tytuł wykresu
        plt.xlabel("PC1")  # etykieta osi X
        plt.ylabel("PC2")  # etykieta osi Y
        plt.savefig("pca2d.png", bbox_inches="tight", dpi=150)  # zapisz wykres PCA do pliku
        plt.close()  # zamknij figurę

    # ============================================================
    # 7. Split danych
    # ============================================================

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )  # spróbuj podzielić dane zachowując proporcje klas (stratify)
    except ValueError:
        print("\nstratify=y nie zadziałało — używam zwykłego podziału.")  # komunikat, jeśli stratify się nie powiedzie
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )  # zwykły podział danych bez stratify

    # ============================================================
    # 8. Decision Tree
    # ============================================================

    dt_model = Pipeline([
        ("preprocess", preprocess),  # najpierw preprocessing
        ("clf", DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42))  # potem klasyfikator drzewa
    ])

    dt_model.fit(X_train, y_train)  # naucz model drzewa na danych treningowych
    dt_pred = dt_model.predict(X_test)  # przewiduj etykiety dla danych testowych

    print("\nDECISION TREE")  # nagłówek sekcji wyników drzewa decyzyjnego
    print(classification_report(y_test, dt_pred))  # wypisz szczegółowy raport (precision/recall/f1)
    print("Confusion matrix:\n", confusion_matrix(y_test, dt_pred))  # wypisz macierz pomyłek

    # ============================================================
    # 9. SVM różne kernele
    # ============================================================

    kernels = ["linear", "rbf", "poly", "sigmoid"]  # lista kernelów, które sprawdzimy dla SVM
    svm_results = {}  # słownik na wyniki dla każdego kernela

    for k in kernels:
        svm_pipe = Pipeline([
            ("preprocess", preprocess),  # preprocessing jako pierwszy krok
            ("clf", SVC(kernel=k, gamma="scale", random_state=42))  # SVM z danym kernelem
        ])
        svm_pipe.fit(X_train, y_train)  # trenuj SVM na danych treningowych
        pred = svm_pipe.predict(X_test)  # przewiduj dla danych testowych

        svm_results[k] = {
            "accuracy": accuracy_score(y_test, pred),  # dokładność predykcji
            "precision": precision_score(y_test, pred, average="macro", zero_division=0),  # precyzja (średnia makro)
            "recall": recall_score(y_test, pred, average="macro", zero_division=0),  # recall (średnia makro)
            "f1": f1_score(y_test, pred, average="macro", zero_division=0)  # f1-score (średnia makro)
        }  # zapisz metryki do słownika dla danego kernela

    print("\nSVM – porównanie kernelów")  # nagłówek dla porównania kernelów SVM
    for k, res in svm_results.items():
        print(f"\nKernel: {k}")  # wypisz nazwę kernela
        for m, v in res.items():
            print(f"  {m}: {v:.4f}")  # wypisz każdą metrykę z formatowaniem do 4 miejsc po przecinku

    # ============================================================
    # 10. Predykcja przykładowa
    # ============================================================

    print("\nPrzykładowa predykcja")  # nagłówek sekcji z przykładową predykcją
    if len(X_train) > 0:
        example = X_train.iloc[[0]]  # wybierz pierwszy przykład z danych treningowych jako próbkę
        print("Dane wejściowe:", example.to_dict(orient='records')[0])  # pokaż dane wejściowe jako słownik
        print("Decision Tree", dt_model.predict(example))  # pokaż predykcję drzewa dla próbki
        svm_rbf = Pipeline([
            ("preprocess", preprocess),  # preprocessing dla SVM RBF
            ("clf", SVC(kernel="rbf", gamma="scale", random_state=42))  # SVM z kernelem rbf
        ]).fit(X_train, y_train)  # od razu trenuj model RBF na danych treningowych
        print("SVM (RBF)", svm_rbf.predict(example))  # pokaż predykcję SVM RBF dla próbki
    else:
        print("Brak danych treningowych — nie można zrobić predykcji.")  # informacja, gdy brak danych do treningu

if __name__ == "__main__":
    main()  # wywołaj funkcję main tylko jeśli skrypt jest uruchamiany bezpośrednio
