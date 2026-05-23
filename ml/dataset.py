"""
Chargement du dataset d'images pour la classification Sentinel.

Convention de dossiers (standard PyTorch ImageFolder) :
    ml/data/
        train/human/      → images de personnes
        train/non_human/
        val/human/
        val/non_human/

Chaque sous-dossier = une classe. Le nom du dossier devient le label.
Cette organisation rend le passage au multi-classes trivial : il suffira
d'ajouter des dossiers (animal/, vehicle/) sans toucher au code.

Auteur : Léo
Projet : Sentinel — classification de cibles radar
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets


def build_dataset(data_dir: Path, transform=None) -> datasets.ImageFolder:
    """
    Construit un dataset PyTorch à partir d'un dossier organisé par classes.

    Args:
        data_dir: dossier contenant un sous-dossier par classe
                  (ex: ml/data/train/ contenant human/ et non_human/).
        transform: transformations appliquées à chaque image. None pour l'instant.

    Returns:
        Un ImageFolder : objet qui sait lire les images et leur label.
    """
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    return dataset


if __name__ == "__main__":
    # Test : charge le dossier d'entraînement et affiche ce que PyTorch a compris.
    train_dir = Path("ml/data/train")
    ds = build_dataset(train_dir)
    print(f"Classes trouvées : {ds.classes}")
    print(f"Mapping classe → label : {ds.class_to_idx}")
    print(f"Nombre total d'images : {len(ds)}")