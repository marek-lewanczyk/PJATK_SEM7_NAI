from __future__ import annotations  # umożliwia używanie adnotacji typów jako stringi

import torch  # główna biblioteka PyTorch
import torch.nn as nn  # moduły sieci neuronowych


# ======================================================
# Punkt 1: MLP dla danych tabelarycznych
# ======================================================

class MLP(nn.Module):
    """
    Wielowarstwowy perceptron (MLP) do klasyfikacji danych tabelarycznych.

    Architektura:
    - warstwa wejściowa -> warstwa ukryta
    - funkcja aktywacji ReLU
    - dropout jako regularizacja
    - druga warstwa ukryta
    - warstwa wyjściowa dopasowana do liczby klas
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Inicjalizacja modelu MLP.

        Parametry:
            input_dim (int): liczba cech wejściowych
            num_classes (int): liczba klas wyjściowych
            hidden (int): liczba neuronów w warstwie ukrytej
            dropout (float): współczynnik dropout
        """

        # inicjalizacja klasy bazowej nn.Module
        super().__init__()

        # definicja architektury sieci
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),  # warstwa wejściowa
            nn.ReLU(),  # nieliniowość
            nn.Dropout(dropout),  # regularyzacja

            nn.Linear(hidden, hidden),  # druga warstwa ukryta
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, num_classes),  # warstwa wyjściowa
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przejście w przód (forward pass).

        Parametry:
            x (torch.Tensor): tensor wejściowy

        Zwraca:
            torch.Tensor: logity klas
        """
        return self.net(x)


# ======================================================
# Punkt 2: CNN dla CIFAR-10
# ======================================================

class CifarCnnSmall(nn.Module):
    """
    Konwolucyjna sieć neuronowa (CNN) dla zbioru CIFAR-10.

    Model:
    - trzy bloki konwolucyjne
    - pooling zmniejszający rozmiar obrazu
    - klasyfikator w pełni połączony
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Inicjalizacja modelu CNN dla CIFAR-10.

        Parametry:
            num_classes (int): liczba klas (domyślnie 10)
        """

        super().__init__()

        # część ekstrakcji cech (feature extractor)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # wejście: 3 kanały RGB
            nn.ReLU(),
            nn.MaxPool2d(2),  # redukcja: 32x32 -> 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )

        # część klasyfikująca
        self.classifier = nn.Sequential(
            nn.Flatten(),  # spłaszczenie map cech
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),  # warstwa wyjściowa
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przejście w przód (forward pass).

        Parametry:
            x (torch.Tensor): batch obrazów CIFAR-10

        Zwraca:
            torch.Tensor: logity klas
        """

        x = self.features(x)  # ekstrakcja cech
        return self.classifier(x)  # klasyfikacja


# ======================================================
# Punkt 3: Fashion-MNIST – modele small / large
# ======================================================

class FashionSmall(nn.Module):
    """
    Mała sieć CNN dla Fashion-MNIST.

    Model bazowy:
    - szybki trening
    - mniejsza liczba parametrów
    - punkt odniesienia dla większej architektury
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Inicjalizacja małego modelu CNN.

        Parametry:
            num_classes (int): liczba klas (domyślnie 10)
        """

        super().__init__()

        # pełna architektura w jednym bloku
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # wejście: 1 kanał (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przejście w przód (forward pass).

        Parametry:
            x (torch.Tensor): batch obrazów Fashion-MNIST

        Zwraca:
            torch.Tensor: logity klas
        """
        return self.net(x)


class FashionLarge(nn.Module):
    """
    Rozszerzona sieć CNN dla Fashion-MNIST.

    Cechy:
    - większa liczba filtrów
    - głębsza architektura
    - intensywniejsza regularizacja (dropout)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Inicjalizacja dużego modelu CNN.

        Parametry:
            num_classes (int): liczba klas (domyślnie 10)
        """

        super().__init__()

        # ekstrakcja cech
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Dropout(0.2),
        )

        # klasyfikator
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przejście w przód (forward pass).

        Parametry:
            x (torch.Tensor): batch obrazów Fashion-MNIST

        Zwraca:
            torch.Tensor: logity klas
        """

        x = self.features(x)  # ekstrakcja cech
        return self.classifier(x)  # klasyfikacja