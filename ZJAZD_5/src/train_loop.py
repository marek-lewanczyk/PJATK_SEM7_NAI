from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from .nn_utils import Logger, ensure_dir


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * x.size(0)
        n += x.size(0)

    return running / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "f1_macro": float(f1)}


@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        yt.append(y.numpy())
        yp.append(pred)
    return np.concatenate(yt), np.concatenate(yp)


def save_confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    ensure_dir(str(out_path).rsplit("/", 1)[0] if "/" in out_path else ".")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    logger: Logger,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    best_metrics = {}

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, test_loader, device)

        logger.log(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | "
                   f"test_acc={metrics['accuracy']:.4f} | test_f1_macro={metrics['f1_macro']:.4f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_metrics = metrics

    logger.log(f"BEST | acc={best_metrics.get('accuracy', 0):.4f} | f1_macro={best_metrics.get('f1_macro', 0):.4f}")
    return best_metrics