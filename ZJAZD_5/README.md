# Klasyfikacja danych – Sieci neuronowe (MLP, CNN)

**Autorzy:**  
Marek Lewańczyk (s29420)  
Katarzyna Kasperek (s27553)

---

## 1. Opis projektu

Projekt prezentuje zastosowanie **sztucznych sieci neuronowych** do problemów klasyfikacji
danych różnego typu: tabelarycznych, obrazowych oraz tekstowych.
Całość została zrealizowana w frameworku **PyTorch**, z wykorzystaniem
narzędzi ekosystemu **scikit-learn** do preprocessingu i modeli bazowych.

Celem projektu było:
- zapoznanie się z podstawami sieci neuronowych,
- porównanie klasycznych metod uczenia maszynowego z sieciami neuronowymi,
- analiza wpływu architektury sieci na jakość klasyfikacji,
- realizacja autorskiego przypadku użycia („zaskocz mnie”).

---

## 2. Zrealizowane punkty zadania

W ramach projektu zrealizowano następujące eksperymenty:

### Punkt 1 – Dane tabelaryczne (Titanic / Telco Customer Churn)
- preprocessing danych (uzupełnianie braków, kodowanie zmiennych kategorycznych),
- porównanie:
  - **baseline:** regresja logistyczna (scikit-learn),
  - **model neuronowy:** MLP (Multi-Layer Perceptron),
- ewaluacja jakości klasyfikacji (accuracy, F1-macro),
- generowanie macierzy pomyłek (confusion matrix).

### Punkt 2 – Klasyfikacja obrazów CIFAR-10
- wykorzystanie konwolucyjnej sieci neuronowej (CNN),
- augmentacja danych (losowe przycięcia, odbicia),
- klasyfikacja 10 klas obiektów (np. samolot, samochód, zwierzęta).

### Punkt 3 – Fashion-MNIST
- klasyfikacja obrazów ubrań,
- porównanie dwóch architektur:
  - **FashionSmall** – mniejsza sieć CNN,
  - **FashionLarge** – większa sieć CNN,
- analiza wpływu rozmiaru sieci na jakość klasyfikacji,
- opcjonalne generowanie confusion matrix.

### Punkt 4 – Autorski przypadek użycia (Customer Churn)
- predykcja cechy użytkownika na zbiorze **Telco Customer Churn**,
- preprocessing danych tabelarycznych (One-Hot Encoding),
- klasyfikacja z wykorzystaniem sieci MLP.

---

## 3. Struktura projektu

```
├── data/                # Zbiory danych (CSV, CIFAR-10, Fashion-MNIST)
├── src/
│   ├── run.py            # Główny plik uruchamiający eksperymenty
│   ├── models.py         # Definicje sieci neuronowych (MLP, CNN)
│   ├── datasets.py       # Loadery danych
│   ├── train_loop.py     # Pętla treningowa i ewaluacja
│   ├── baseline_sklearn.py
│   ├── preprocess_titanic.py
│   ├── preprocess_telco.py
│   └── nn_utils.py       # Narzędzia pomocnicze (seed, logger)
├── logs/                 # Logi treningu
├── figures/              # Confusion matrix
├── results/              # Zapis metryk (JSONL)
└── README.md
```

---

## 4. Wymagania

- Python 3.9+
- PyTorch
- scikit-learn
- pandas
- numpy
- torchvision
- matplotlib
- seaborn

Instalacja (przykładowo):
```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn
```

## 5. Uruchomienie projektu

### Punkt 1
```bash
python3 -m src.run point1_tabular \
  --csv_path data/titanic.csv \
  --target_col Survived \
  --epochs 50 \
  --log_path logs/01_point1_tabular.log
```

### Punkt 2
```bash
python -m src.run point2_cifar10 \
  --epochs 15 \
  --log_path logs/02_point2_cifar10.log
```

### Punkt 3
```bash
python -m src.run point3_fashion \
  --model_size small \
  --epochs 10 \
  --log_path logs/03_fashion_small.log
```

### Punkt 4
```bash
python -m src.run point1_tabular \
  --csv_path data/TelcoCustomerChurn.csv \
  --target_col SeniorCitizen \
  --epochs 40 \
  --log_path logs/04_telco_churn.log
```

---

## 6. Wyniki i omówienie

### Punkt 1 – Titanic
- Regresja logistyczna osiągnęła poprawne, lecz ograniczone wyniki.
- Sieć MLP uzyskała wyższą dokładność oraz lepszy F1-score,
co wskazuje na lepsze modelowanie zależności nieliniowych.
- Confusion matrix pozwala zauważyć poprawę w klasyfikacji klasy mniejszościowej.

### Punkt 2 – CIFAR-10
- CNN skutecznie nauczyła się cech wizualnych obrazów.
- Największe błędy pojawiały się pomiędzy wizualnie podobnymi klasami (np. kot vs pies).

### Punkt 3 – Fashion-MNIST
- Większa sieć (FashionLarge) osiągnęła lepsze wyniki kosztem dłuższego treningu.
- Pokazuje to kompromis pomiędzy złożonością modelu a czasem uczenia.

### Punkt 4 – Telco Customer Churn
- Model MLP poprawnie identyfikował wzorce w danych klientów.
- Przypadek użycia pokazuje praktyczne zastosowanie sieci neuronowych w analizie danych biznesowych.

---

## 7. Wnioski końcowe
- Sieci neuronowe oferują większą elastyczność niż klasyczne modele,
szczególnie dla danych nieliniowych i obrazowych.
- W przypadku prostych danych tabelarycznych różnice są mniejsze,
ale nadal zauważalne.
- Architektura i rozmiar sieci mają istotny wpływ na jakość klasyfikacji.
- Projekt potwierdza uniwersalność sieci neuronowych w różnych domenach danych.