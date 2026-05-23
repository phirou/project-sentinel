"""
Boucle d'entraînement du CNN Sentinel (classification humain / non-humain).

Pipeline : charge les datasets train/val → entraîne sur N epochs →
mesure loss et accuracy à chaque epoch → sauvegarde le meilleur modèle.

Les 4 étapes du cœur de l'entraînement (forward, loss, backward, step)
sont la version PyTorch de la rétropropagation : autograd calcule les
gradients automatiquement via loss.backward().

Lancement :
    python -m ml.train

Auteur : Léo
Projet : Sentinel — classification de cibles radar
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import build_dataset
from ml.transforms import build_train_transform, build_val_transform
from ml.model import SentinelCNN


def get_device() -> torch.device:
    """Choisit le meilleur accélérateur dispo : MPS (Mac M-series) sinon CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Entraîne le modèle sur un passage complet du dataset (1 epoch).

    Returns:
        (loss moyenne, accuracy) sur l'epoch.
    """
    model.train()                      # mode entraînement : Dropout actif
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # On envoie les données sur le même device que le modèle (MPS/CPU).
        images, labels = images.to(device), labels.to(device)

        # --- Les 4 étapes du cœur de l'entraînement ---
        optimizer.zero_grad()                  # 0. remet les gradients à zéro
        outputs = model(images)                # 1. forward : prédictions
        loss = criterion(outputs, labels)      # 2. loss : mesure l'erreur
        loss.backward()                        # 3. backward : autograd → gradients
        optimizer.step()                       # 4. step : ajuste les poids

        # --- Statistiques ---
        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)      # classe prédite = score max
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Évalue le modèle sur le set de validation (sans apprentissage).

    Le décorateur @torch.no_grad() désactive le calcul des gradients :
    on ne fait que mesurer, pas apprendre → plus rapide, moins de mémoire.

    Returns:
        (loss moyenne, accuracy) sur la validation.
    """
    model.eval()                       # mode éval : Dropout désactivé
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)              # forward seulement
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main() -> None:
    # --- Hyperparamètres (à externaliser dans config.yaml plus tard) ---
    data_root = Path("ml/data")
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    device = get_device()
    print(f"Device : {device}")

    # --- Datasets et loaders ---
    train_ds = build_dataset(data_root / "train", transform=build_train_transform())
    val_ds = build_dataset(data_root / "val", transform=build_val_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train : {len(train_ds)} images | Val : {len(val_ds)} images")
    print(f"Classes : {train_ds.classes}")

    # --- Modèle, loss, optimizer ---
    model = SentinelCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Boucle d'entraînement ---
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}"
        )

        # Sauvegarde le meilleur modèle (selon l'accuracy de validation).
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "ml/sentinel_cnn_best.pth")
            print(f"  ↳ nouveau meilleur modèle sauvegardé (val acc {val_acc:.3f})")

    print(f"\nEntraînement terminé. Meilleure val acc : {best_val_acc:.3f}")


if __name__ == "__main__":
    main()