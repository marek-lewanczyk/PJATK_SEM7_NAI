"""
Projekt: Klasyfikacja danych – Sieci neuronowe (MLP, CNN)
Autorzy: Marek Lewańczyk s29420, Katarzyna Kasperek s27553

Opis:
  Projekt realizuje kompletny pipeline uczenia maszynowego z wykorzystaniem
  sztucznych sieci neuronowych w frameworku PyTorch. Obejmuje wczytanie danych,
  preprocessing (w tym kodowanie zmiennych kategorycznych i standaryzację),
  trening modeli neuronowych oraz ewaluację jakości klasyfikacji.

  W ramach projektu zrealizowano:
  - porównanie klasycznego podejścia (regresja logistyczna) z siecią MLP
    na zbiorze Titanic,
  - klasyfikację obrazów z wykorzystaniem CNN na zbiorach CIFAR-10
    oraz Fashion-MNIST,
  - porównanie dwóch rozmiarów sieci neuronowych (small vs large),
  - generowanie confusion matrix,
  - autorski przypadek użycia: predykcja odejścia klienta (Customer Churn)
    na zbiorze Telco Customer Churn.

Instrukcja użycia:
  1. Umieść wybrany zbiór danych w katalogu /data.
  2. Wybierz odpowiedni tryb uruchomienia (punkt 1–4) poprzez argument CLI.
  3. Uruchom projekt poleceniem:
     python -m src.run <task> [parametry]
  4. Logi treningu zapisywane są w katalogu /logs,
     a wygenerowane confusion matrix w katalogu /figures.

Przykładowe zadania:
  - point1_tabular – klasyfikacja danych tabelarycznych (Titanic, Telco Churn),
  - point2_cifar10 – klasyfikacja obrazów CIFAR-10,
  - point3_fashion – klasyfikacja Fashion-MNIST (dwa rozmiary sieci),
  - point4_text – klasyfikacja danych tekstowych (TF-IDF + MLP).

Referencje:
  - PyTorch: https://pytorch.org/
  - Scikit-learn: https://scikit-learn.org/
  - Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
  - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .nn_utils import set_seed, get_device, Logger, save_jsonl, ensure_dir
from .datasets import (
    TabularConfig, load_tabular_csv, tabular_to_torch_loaders,
    cifar10_loaders, fashion_mnist_loaders,
    TextCsvConfig, load_text_csv
)
from .models import MLP, CifarCnnSmall, FashionSmall, FashionLarge
from .train_loop import TrainConfig, fit, get_predictions, save_confusion_matrix_png
from .baseline_sklearn import BaselineConfig, run_baseline_logreg
from .preprocess_titanic import preprocess_titanic
from .preprocess_telco import preprocess_telco


def run_point1_tabular(args: argparse.Namespace) -> None:
    """
    Punkt 1:
    - baseline sklearn (LogReg)
    - NN (MLP)
    """
    log = Logger(args.log_path)
    device = get_device()
    log.log(f"[P1] device={device}")

    # cfg = TabularConfig(csv_path=Path(args.csv_path), target_col=args.target_col, test_ratio=args.test_ratio, seed=args.seed)
    # X_train, X_test, y_train, y_test = load_tabular_csv(cfg)
    # log.log(f"[P1] loaded CSV: X_train={X_train.shape}, X_test={X_test.shape}")
    # ===== Dataset-specific preprocessing =====
    csv_lower = args.csv_path.lower()

    if "titanic" in csv_lower:
        X, y = preprocess_titanic(args.csv_path)
        dataset_name = "Titanic"
        log.log("[P1] Using Titanic-specific preprocessing")

    elif "telco" in csv_lower:
        X, y = preprocess_telco(args.csv_path)
        dataset_name = "Telco Churn"
        log.log("[P4] Using Telco-specific preprocessing")

    else:
        cfg = TabularConfig(
            csv_path=Path(args.csv_path),
            target_col=args.target_col,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        X_train, X_test, y_train, y_test = load_tabular_csv(cfg)
        dataset_name = "Generic tabular"
        log.log("[P1] Using generic tabular loader")

    if "titanic" in csv_lower or "telco" in csv_lower:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)

        split = int(len(idx) * (1.0 - args.test_ratio))
        train_idx, test_idx = idx[:split], idx[split:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        log.log(f"[{dataset_name}] X_train={X_train.shape}, X_test={X_test.shape}")

    # Baseline
    baseline_metrics = run_baseline_logreg(X_train, y_train, X_test, y_test, BaselineConfig())
    log.log(f"[P1] baseline(LogReg) acc={baseline_metrics['accuracy']:.4f}, f1_macro={baseline_metrics['f1_macro']:.4f}")

    # NN loaders (tu już mapujemy klasy)
    # Standaryzacja w NN: najprościej policzyć mean/std z train i zastosować na oba
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    train_loader, test_loader, input_dim, num_classes = tabular_to_torch_loaders(
        X_train_s.astype(np.float32), X_test_s.astype(np.float32), y_train, y_test, batch_size=args.batch_size
    )

    model = MLP(input_dim=input_dim, num_classes=num_classes, hidden=args.hidden, dropout=args.dropout)
    nn_metrics = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        logger=log,
        cfg=TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay),
    )
    log.log(f"[P1] nn(MLP) acc={nn_metrics['accuracy']:.4f}, f1_macro={nn_metrics['f1_macro']:.4f}")

    # ===== Confusion matrix (Titanic – MLP) =====
    cm_path = f"figures/confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png"

    y_true, y_pred = get_predictions(model, test_loader, device)
    save_confusion_matrix_png(
        y_true,
        y_pred,
        cm_path,
        title=f"{dataset_name} – MLP"
    )
    log.log(f"[P] saved confusion matrix: {cm_path}")

    save_jsonl("results/metrics.jsonl", {
        "task": "point1_tabular",
        "baseline": baseline_metrics,
        "nn": nn_metrics,
        "csv": str(args.csv_path),
        "target_col": args.target_col,
    })


def run_point2_cifar10(args: argparse.Namespace) -> None:
    log = Logger(args.log_path)
    device = get_device()
    log.log(f"[P2] device={device}")

    train_loader, test_loader, num_classes = cifar10_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    model = CifarCnnSmall(num_classes=num_classes)

    metrics = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        logger=log,
        cfg=TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay),
    )
    save_jsonl("results/metrics.jsonl", {"task": "point2_cifar10", "metrics": metrics})


def run_point3_fashion(args: argparse.Namespace) -> None:
    log = Logger(args.log_path)
    device = get_device()
    log.log(f"[P3] device={device}")

    train_loader, test_loader, num_classes = fashion_mnist_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    if args.model_size == "small":
        model = FashionSmall(num_classes=num_classes)
    elif args.model_size == "large":
        model = FashionLarge(num_classes=num_classes)
    else:
        raise ValueError("model_size must be 'small' or 'large'")

    metrics = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        logger=log,
        cfg=TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay),
    )

    # confusion matrix (wymóg: dla jednego z punktów)
    if args.save_confmat:
        y_true, y_pred = get_predictions(model, test_loader, device)
        out_path = args.save_confmat
        save_confusion_matrix_png(y_true, y_pred, out_path, title=f"Fashion-MNIST ({args.model_size})")
        log.log(f"[P3] saved confusion matrix: {out_path}")

    save_jsonl("results/metrics.jsonl", {"task": "point3_fashion", "model_size": args.model_size, "metrics": metrics})


def run_point4_text(args: argparse.Namespace) -> None:
    """
    Punkt 4 (zaskoczenie): tekst -> TF-IDF -> MLP.
    Jedna technologia NN zostaje: PyTorch (MLP).
    """
    log = Logger(args.log_path)
    device = get_device()
    log.log(f"[P4] device={device}")

    cfg = TextCsvConfig(csv_path=Path(args.csv_path), text_col=args.text_col, target_col=args.target_col,
                        test_ratio=args.test_ratio, seed=args.seed)
    X_train_txt, X_test_txt, y_train, y_test = load_text_csv(cfg)
    log.log(f"[P4] loaded text CSV: train={len(X_train_txt)}, test={len(X_test_txt)}")

    # TF-IDF jako wektory wejściowe
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_txt).astype(np.float32)
    X_test = vectorizer.transform(X_test_txt).astype(np.float32)

    # do torch (dense) — dla małych max_features ok; na laboratoria wystarczy
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # mapowanie klas
    classes, y_train_idx = np.unique(y_train, return_inverse=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test_idx = np.array([class_to_idx[v] for v in y_test], dtype=np.int64)

    # loaders
    train_loader, test_loader, input_dim, num_classes = tabular_to_torch_loaders(
        X_train, X_test, y_train_idx, y_test_idx, batch_size=args.batch_size
    )

    model = MLP(input_dim=input_dim, num_classes=num_classes, hidden=args.hidden, dropout=args.dropout)
    metrics = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        logger=log,
        cfg=TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay),
    )

    save_jsonl("results/metrics.jsonl", {
        "task": "point4_text",
        "metrics": metrics,
        "csv": str(args.csv_path),
        "text_col": args.text_col,
        "target_col": args.target_col,
        "max_features": args.max_features
    })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)

    sub = p.add_subparsers(dest="task", required=True)

    # Punkt 1
    p1 = sub.add_parser("point1_tabular")
    p1.add_argument("--csv_path", required=True)
    p1.add_argument("--target_col", required=True)
    p1.add_argument("--test_ratio", type=float, default=0.2)
    p1.add_argument("--epochs", type=int, default=50)
    p1.add_argument("--batch_size", type=int, default=64)
    p1.add_argument("--lr", type=float, default=1e-3)
    p1.add_argument("--weight_decay", type=float, default=1e-4)
    p1.add_argument("--hidden", type=int, default=128)
    p1.add_argument("--dropout", type=float, default=0.1)
    p1.add_argument("--log_path", default="logs/01_point1_tabular.log")

    # Punkt 2
    p2 = sub.add_parser("point2_cifar10")
    p2.add_argument("--epochs", type=int, default=15)
    p2.add_argument("--batch_size", type=int, default=128)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--weight_decay", type=float, default=1e-4)
    p2.add_argument("--num_workers", type=int, default=2)
    p2.add_argument("--log_path", default="logs/02_point2_cifar10.log")

    # Punkt 3
    p3 = sub.add_parser("point3_fashion")
    p3.add_argument("--model_size", choices=["small", "large"], required=True)
    p3.add_argument("--epochs", type=int, default=10)
    p3.add_argument("--batch_size", type=int, default=128)
    p3.add_argument("--lr", type=float, default=1e-3)
    p3.add_argument("--weight_decay", type=float, default=1e-4)
    p3.add_argument("--num_workers", type=int, default=2)
    p3.add_argument("--save_confmat", default="")  # np. figures/confusion_matrix.png
    p3.add_argument("--log_path", default="logs/03_point3_fashion.log")

    # Punkt 4
    p4 = sub.add_parser("point4_text")
    p4.add_argument("--csv_path", required=True)
    p4.add_argument("--text_col", required=True)
    p4.add_argument("--target_col", required=True)
    p4.add_argument("--test_ratio", type=float, default=0.2)
    p4.add_argument("--epochs", type=int, default=20)
    p4.add_argument("--batch_size", type=int, default=64)
    p4.add_argument("--lr", type=float, default=1e-3)
    p4.add_argument("--weight_decay", type=float, default=1e-4)
    p4.add_argument("--hidden", type=int, default=128)
    p4.add_argument("--dropout", type=float, default=0.1)
    p4.add_argument("--max_features", type=int, default=5000)
    p4.add_argument("--log_path", default="logs/04_point4_text.log")

    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    ensure_dir("results")

    if args.task == "point1_tabular":
        run_point1_tabular(args)
    elif args.task == "point2_cifar10":
        run_point2_cifar10(args)
    elif args.task == "point3_fashion":
        run_point3_fashion(args)
    elif args.task == "point4_text":
        run_point4_text(args)
    else:
        raise ValueError("Unknown task")


if __name__ == "__main__":
    main()