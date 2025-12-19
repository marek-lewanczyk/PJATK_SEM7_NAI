import pandas as pd  # obsługa danych tabelarycznych
import numpy as np  # obliczenia numeryczne


def preprocess_telco(csv_path: str):
    """
    Realizuje preprocessing zbioru Telco Customer Churn.

    Funkcja przygotowuje dane do uczenia modeli klasyfikacyjnych
    (baseline oraz sieć neuronowa MLP).

    Kroki preprocessingu:
    - wczytanie danych z CSV,
    - mapowanie kolumny docelowej (Churn) na wartości binarne,
    - usunięcie identyfikatora klienta,
    - konwersja kolumny TotalCharges na wartości numeryczne,
    - uzupełnienie braków danych medianą,
    - kodowanie zmiennych kategorycznych metodą one-hot encoding.

    Parametry:
        csv_path (str): ścieżka do pliku CSV z danymi Telco

    Zwraca:
        Tuple[np.ndarray, np.ndarray]:
            X (np.ndarray): macierz cech wejściowych (float32)
            y (np.ndarray): wektor etykiet klas (0 – brak churn, 1 – churn)
    """

    # wczytanie danych z pliku CSV
    df = pd.read_csv(csv_path)

    # mapowanie kolumny docelowej "Churn" na wartości binarne
    # Yes -> 1 (klient odchodzi), No -> 0 (klient pozostaje)
    y = df["Churn"].map({"Yes": 1, "No": 0}).to_numpy()

    # usunięcie kolumny identyfikatora oraz kolumny docelowej
    df = df.drop(columns=["customerID", "Churn"])

    # kolumna TotalCharges bywa typu string – konwersja na float
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",  # błędy konwersji zamieniane na NaN
    )

    # uzupełnienie brakujących wartości medianą kolumny
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # kodowanie zmiennych kategorycznych (one-hot encoding)
    # drop_first=True zapobiega nadmiarowości cech
    df = pd.get_dummies(df, drop_first=True)

    # konwersja DataFrame do macierzy NumPy typu float32
    X = df.to_numpy(dtype=np.float32)

    return X, y