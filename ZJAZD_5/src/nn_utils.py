from __future__ import annotations  # umożliwia używanie adnotacji typów jako stringi

import json  # zapis wyników w formacie JSON
import os  # operacje systemowe
import random  # generatory losowe Pythona
import time  # obsługa znaczników czasu
from dataclasses import dataclass  # definicja klas danych
from pathlib import Path  # obsługa ścieżek plików
from typing import Any, Dict, Optional  # typowanie

import numpy as np  # obliczenia numeryczne
import torch  # framework PyTorch


def set_seed(seed: int) -> None:
    """
    Ustawia ziarno losowości dla wszystkich używanych bibliotek.

    Funkcja zapewnia powtarzalność eksperymentów poprzez
    ustawienie seedów dla:
    - standardowego generatora losowego Pythona,
    - NumPy,
    - PyTorch (CPU i GPU).

    Dodatkowo wymusza deterministyczne zachowanie cudnn
    kosztem wydajności (wystarczające na potrzeby laboratoriów).

    Parametry:
        seed (int): wartość ziarna losowości
    """

    # seed dla biblioteki random
    random.seed(seed)

    # seed dla NumPy
    np.random.seed(seed)

    # seed dla PyTorch (CPU)
    torch.manual_seed(seed)

    # seed dla PyTorch (GPU, jeśli dostępne)
    torch.cuda.manual_seed_all(seed)

    # wymuszenie deterministycznego działania cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    """
    Tworzy katalog, jeśli nie istnieje.

    Funkcja jest bezpieczna:
    - nie zgłasza błędu, jeśli katalog już istnieje,
    - tworzy również katalogi nadrzędne.

    Parametry:
        path (str | Path): ścieżka do katalogu
    """

    # konwersja do obiektu Path i utworzenie katalogu
    Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class RunPaths:
    """
    Przechowuje ścieżki używane w trakcie uruchamiania eksperymentów.

    Atrybuty:
        logs_dir (Path): katalog na pliki logów
        figures_dir (Path): katalog na wykresy i confusion matrix
    """

    logs_dir: Path  # katalog logów
    figures_dir: Path  # katalog wykresów

    def __post_init__(self) -> None:
        """
        Metoda wywoływana po inicjalizacji obiektu.

        Zapewnia istnienie katalogów na logi i wykresy.
        """

        ensure_dir(self.logs_dir)
        ensure_dir(self.figures_dir)


class Logger:
    """
    Prosty logger zapisujący komunikaty:
    - do standardowego wyjścia (stdout),
    - do pliku logów.

    Rozwiązanie użyte celowo zamiast logging,
    aby logi były czytelne i łatwe do oceny w repozytorium.
    """

    def __init__(self, log_path: str | Path) -> None:
        """
        Inicjalizacja loggera.

        Parametry:
            log_path (str | Path): ścieżka do pliku logów
        """

        # zapis ścieżki do pliku logów
        self.log_path = Path(log_path)

        # upewnienie się, że katalog logów istnieje
        ensure_dir(self.log_path.parent)

    def log(self, msg: str) -> None:
        """
        Zapisuje pojedynczą linię logu z aktualnym znacznikiem czasu.

        Parametry:
            msg (str): treść komunikatu
        """

        # utworzenie linii logu z timestampem
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"

        # wypisanie do konsoli
        print(line)

        # dopisanie do pliku logów
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def get_device() -> torch.device:
    """
    Wybiera urządzenie obliczeniowe dla PyTorch.

    Zwraca:
        torch.device:
            - 'cuda' jeśli GPU jest dostępne,
            - 'cpu' w przeciwnym wypadku.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    """
    Zapisuje pojedynczy rekord w formacie JSON Lines (JSONL).

    Każde wywołanie dopisuje nową linię do pliku,
    co ułatwia zapisywanie wyników wielu eksperymentów.

    Parametry:
        path (str | Path): ścieżka do pliku JSONL
        record (Dict[str, Any]): dane do zapisania
    """

    # konwersja ścieżki do Path
    path = Path(path)

    # upewnienie się, że katalog docelowy istnieje
    ensure_dir(path.parent)

    # zapis rekordu jako pojedynczej linii JSON
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")