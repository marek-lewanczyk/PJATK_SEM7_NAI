import pandas as pd  # obsługa danych tabelarycznych
import numpy as np  # obliczenia numeryczne


def preprocess_titanic(csv_path: str):
    """
    Realizuje preprocessing zbioru danych Titanic.

    Funkcja przygotowuje dane do klasyfikacji przeżycia pasażerów
    z wykorzystaniem modeli klasycznych oraz sieci neuronowych.

    Kroki preprocessingu:
    - wczytanie danych z pliku CSV,
    - wydzielenie kolumny docelowej (Survived),
    - wybór istotnych cech wejściowych,
    - uzupełnienie braków danych,
    - kodowanie zmiennych kategorycznych,
    - konwersja danych do formatu NumPy.

    Parametry:
        csv_path (str): ścieżka do pliku CSV ze zbiorem Titanic

    Zwraca:
        Tuple[np.ndarray, np.ndarray]:
            X (np.ndarray): macierz cech wejściowych (float32)
            y (np.ndarray): wektor etykiet klas (0/1)
    """

    # wczytanie danych z pliku CSV
    df = pd.read_csv(csv_path)

    # wydzielenie kolumny docelowej (0 – nie przeżył, 1 – przeżył)
    y = df["Survived"].to_numpy()

    # wybór istotnych cech wejściowych
    X = df[
        [
            "Pclass",   # klasa podróży
            "Sex",      # płeć
            "Age",      # wiek
            "SibSp",    # liczba rodzeństwa/małżonków
            "Parch",    # liczba rodziców/dzieci
            "Fare",     # cena biletu
            "Embarked", # port zaokrętowania
        ]
    ].copy()

    # uzupełnienie brakujących wartości wieku medianą
    X["Age"] = X["Age"].fillna(X["Age"].median())

    # uzupełnienie brakujących portów zaokrętowania najczęstszą wartością
    X["Embarked"] = X["Embarked"].fillna("S")

    # kodowanie zmiennej płci: male -> 0, female -> 1
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

    # kodowanie zmiennej kategorycznej Embarked (one-hot encoding)
    # drop_first=True zapobiega nadmiarowości cech
    X = pd.get_dummies(X, columns=["Embarked"], drop_first=True)

    # konwersja DataFrame do macierzy NumPy typu float32
    X = X.to_numpy(dtype=np.float32)

    return X, y