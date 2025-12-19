from __future__ import annotations  # umożliwia używanie adnotacji typów jako stringi

from dataclasses import dataclass  # uproszczona definicja klas konfiguracyjnych
from typing import Dict  # typowanie słowników wyników

import numpy as np  # obliczenia numeryczne
from sklearn.metrics import accuracy_score, f1_score  # metryki klasyfikacji
from sklearn.pipeline import Pipeline  # pipeline preprocessing + model
from sklearn.preprocessing import StandardScaler  # standaryzacja cech
from sklearn.linear_model import LogisticRegression  # regresja logistyczna


@dataclass
class BaselineConfig:
    """
    Konfiguracja modelu bazowego (baseline).

    Przechowuje hiperparametry dla regresji logistycznej
    wykorzystywanej jako punkt odniesienia dla sieci neuronowych.

    Atrybuty:
        max_iter (int): maksymalna liczba iteracji algorytmu optymalizacji
    """

    max_iter: int = 2000  # domyślna liczba iteracji


def run_baseline_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: BaselineConfig,
) -> Dict[str, float]:
    """
    Uruchamia klasyczny model bazowy: regresję logistyczną.

    Funkcja realizuje pełny pipeline:
    - mapowanie etykiet klas do zakresu 0..K-1,
    - standaryzację cech,
    - trening regresji logistycznej,
    - ewaluację na zbiorze testowym.

    Parametry:
        X_train (np.ndarray): dane treningowe (cechy)
        y_train (np.ndarray): etykiety treningowe
        X_test (np.ndarray): dane testowe (cechy)
        y_test (np.ndarray): etykiety testowe
        cfg (BaselineConfig): konfiguracja modelu bazowego

    Zwraca:
        Dict[str, float]: słownik z metrykami jakości:
            - accuracy
            - f1_macro
    """

    # mapowanie oryginalnych etykiet klas na indeksy 0..K-1
    classes, y_train_idx = np.unique(y_train, return_inverse=True)

    # słownik mapujący klasę oryginalną -> indeks
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # mapowanie etykiet testowych do tej samej przestrzeni indeksów
    y_test_idx = np.array(
        [class_to_idx[v] for v in y_test],
        dtype=np.int64,
    )

    # definicja pipeline: standaryzacja + regresja logistyczna
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),  # skalowanie cech
            (
                "clf",
                LogisticRegression(
                    max_iter=cfg.max_iter  # maks. liczba iteracji
                ),
            ),
        ]
    )

    # trening modelu na danych treningowych
    pipe.fit(X_train, y_train_idx)

    # predykcja na zbiorze testowym
    pred = pipe.predict(X_test)

    # obliczenie accuracy
    acc = accuracy_score(y_test_idx, pred)

    # obliczenie F1-score (średnia makro)
    f1 = f1_score(y_test_idx, pred, average="macro")

    # zwrot metryk w postaci słownika
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
    }