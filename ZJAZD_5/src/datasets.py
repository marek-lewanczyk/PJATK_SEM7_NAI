from __future__ import annotations  # umożliwia używanie adnotacji typów jako stringi

from dataclasses import dataclass  # definicja prostych klas konfiguracyjnych
from pathlib import Path  # obsługa ścieżek plików
from typing import Tuple, Optional  # typowanie zwracanych wartości

import numpy as np  # obliczenia numeryczne
import pandas as pd  # obsługa danych tabelarycznych
import torch  # framework PyTorch
from torch.utils.data import DataLoader, TensorDataset  # ładowanie danych do PyTorch
from torchvision import datasets, transforms  # wbudowane zbiory danych i transformacje


# =========================
# Punkt 1: dane tabelaryczne (CSV)
# =========================

@dataclass
class TabularConfig:
    """
    Konfiguracja dla danych tabelarycznych w formacie CSV.

    Atrybuty:
        csv_path (Path): ścieżka do pliku CSV
        target_col (str): nazwa kolumny docelowej (etykiety)
        test_ratio (float): proporcja danych testowych
        seed (int): ziarno losowości
    """

    csv_path: Path  # ścieżka do pliku CSV
    target_col: str  # kolumna docelowa
    test_ratio: float = 0.2  # udział danych testowych
    seed: int = 42  # seed losowości


def load_tabular_csv(cfg: TabularConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wczytuje dane tabelaryczne z pliku CSV i dzieli je na zbiór treningowy oraz testowy.

    Założenia:
    - wszystkie cechy są numeryczne lub zostały wcześniej zakodowane,
    - kolumna target_col zawiera etykiety klas.

    Parametry:
        cfg (TabularConfig): konfiguracja loadera CSV

    Zwraca:
        Tuple:
            X_train (np.ndarray): cechy treningowe
            X_test (np.ndarray): cechy testowe
            y_train (np.ndarray): etykiety treningowe
            y_test (np.ndarray): etykiety testowe
    """

    # wczytanie danych z pliku CSV
    df = pd.read_csv(cfg.csv_path)

    # sprawdzenie, czy kolumna docelowa istnieje
    if cfg.target_col not in df.columns:
        raise ValueError(
            f"target_col='{cfg.target_col}' nie istnieje w CSV. Dostępne: {list(df.columns)}"
        )

    # wydzielenie etykiet
    y = df[cfg.target_col].to_numpy()

    # wydzielenie cech i konwersja do float32
    X = df.drop(columns=[cfg.target_col]).to_numpy(dtype=np.float32)

    # inicjalizacja generatora liczb losowych
    rng = np.random.default_rng(cfg.seed)

    # permutacja indeksów
    idx = np.arange(len(df))
    rng.shuffle(idx)

    # obliczenie punktu podziału train/test
    split = int(len(idx) * (1.0 - cfg.test_ratio))

    # indeksy zbiorów
    train_idx, test_idx = idx[:split], idx[split:]

    # podział danych
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def tabular_to_torch_loaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Konwertuje dane tabelaryczne do obiektów DataLoader PyTorch.

    Funkcja:
    - mapuje etykiety klas na zakres 0..K-1,
    - tworzy tensory PyTorch,
    - buduje DataLoader dla treningu i testu.

    Parametry:
        X_train (np.ndarray): cechy treningowe
        X_test (np.ndarray): cechy testowe
        y_train (np.ndarray): etykiety treningowe
        y_test (np.ndarray): etykiety testowe
        batch_size (int): rozmiar batcha

    Zwraca:
        Tuple:
            train_loader (DataLoader)
            test_loader (DataLoader)
            input_dim (int): liczba cech wejściowych
            num_classes (int): liczba klas
    """

    # mapowanie etykiet klas na indeksy 0..K-1
    classes, y_train_idx = np.unique(y_train, return_inverse=True)

    # słownik mapujący oryginalne etykiety na indeksy
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # mapowanie etykiet testowych
    y_test_idx = np.array(
        [class_to_idx[v] for v in y_test],
        dtype=np.int64,
    )

    # konwersja cech do tensorów PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # konwersja etykiet do tensorów PyTorch
    y_train_t = torch.tensor(y_train_idx, dtype=torch.long)
    y_test_t = torch.tensor(y_test_idx, dtype=torch.long)

    # utworzenie datasetów PyTorch
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # DataLoader dla zbioru treningowego
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    # DataLoader dla zbioru testowego
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # liczba cech wejściowych
    input_dim = X_train.shape[1]

    # liczba klas
    num_classes = len(classes)

    return train_loader, test_loader, input_dim, num_classes


# =========================
# Punkt 2: CIFAR-10
# =========================

def cifar10_loaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, int]:
    """
    Tworzy DataLoader dla zbioru CIFAR-10.

    Zawiera:
    - augmentację danych treningowych,
    - normalizację obrazów,
    - automatyczne pobieranie zbioru danych.

    Parametry:
        batch_size (int): rozmiar batcha
        num_workers (int): liczba wątków do ładowania danych

    Zwraca:
        Tuple:
            train_loader (DataLoader)
            test_loader (DataLoader)
            num_classes (int): liczba klas (10)
    """

    # transformacje dla zbioru treningowego
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # losowe odbicie w poziomie
        transforms.RandomCrop(32, padding=4),  # losowe przycięcie
        transforms.ToTensor(),  # konwersja do tensora
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.247, 0.243, 0.261),
        ),
    ])

    # transformacje dla zbioru testowego
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.247, 0.243, 0.261),
        ),
    ])

    # zbiór treningowy CIFAR-10
    train_ds = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=tf_train,
    )

    # zbiór testowy CIFAR-10
    test_ds = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=tf_test,
    )

    # DataLoader treningowy
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # DataLoader testowy
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, 10


# =========================
# Punkt 3: Fashion-MNIST
# =========================

def fashion_mnist_loaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, int]:
    """
    Tworzy DataLoader dla zbioru Fashion-MNIST.

    Parametry:
        batch_size (int): rozmiar batcha
        num_workers (int): liczba wątków

    Zwraca:
        Tuple:
            train_loader (DataLoader)
            test_loader (DataLoader)
            num_classes (int): liczba klas (10)
    """

    # transformacje wspólne dla train i test
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    # zbiór treningowy Fashion-MNIST
    train_ds = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=tf,
    )

    # zbiór testowy Fashion-MNIST
    test_ds = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=tf,
    )

    # DataLoader treningowy
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # DataLoader testowy
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, 10


# =========================
# Punkt 4: dane tekstowe (TF-IDF + MLP)
# =========================

@dataclass
class TextCsvConfig:
    """
    Konfiguracja dla danych tekstowych zapisanych w CSV.

    Atrybuty:
        csv_path (Path): ścieżka do pliku CSV
        text_col (str): kolumna z tekstem
        target_col (str): kolumna z etykietą klasy
        test_ratio (float): udział danych testowych
        seed (int): ziarno losowości
    """

    csv_path: Path
    text_col: str
    target_col: str
    test_ratio: float = 0.2
    seed: int = 42


def load_text_csv(cfg: TextCsvConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wczytuje dane tekstowe z CSV i dzieli je na zbiór treningowy oraz testowy.

    Parametry:
        cfg (TextCsvConfig): konfiguracja loadera danych tekstowych

    Zwraca:
        Tuple:
            X_train (np.ndarray): teksty treningowe
            X_test (np.ndarray): teksty testowe
            y_train (np.ndarray): etykiety treningowe
            y_test (np.ndarray): etykiety testowe
    """

    # wczytanie danych z CSV
    df = pd.read_csv(cfg.csv_path)

    # sprawdzenie wymaganych kolumn
    if cfg.text_col not in df.columns or cfg.target_col not in df.columns:
        raise ValueError(
            f"Brakuje kolumn. Jest: {list(df.columns)}"
        )

    # wydzielenie tekstów i etykiet
    X = df[cfg.text_col].astype(str).to_numpy()
    y = df[cfg.target_col].to_numpy()

    # inicjalizacja generatora losowego
    rng = np.random.default_rng(cfg.seed)

    # permutacja indeksów
    idx = np.arange(len(df))
    rng.shuffle(idx)

    # punkt podziału train/test
    split = int(len(idx) * (1.0 - cfg.test_ratio))

    # indeksy zbiorów
    train_idx, test_idx = idx[:split], idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]