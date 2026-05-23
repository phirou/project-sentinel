"""
Transformations d'images pour la classification Sentinel.

Deux pipelines distincts :
  - ENTRAÎNEMENT : redimensionnement + augmentation (flips, rotations) +
    normalisation. L'augmentation crée de la variété artificielle pour
    réduire le surapprentissage.
  - VALIDATION : redimensionnement + normalisation uniquement. AUCUNE
    augmentation : on veut mesurer la performance réelle, pas déformée.

La taille d'entrée du réseau est fixée par IMAGE_SIZE. Toute image, quelle
que soit sa résolution d'origine, est ramenée à IMAGE_SIZE × IMAGE_SIZE.

Auteur : Léo
Projet : Sentinel — classification de cibles radar
"""

from __future__ import annotations

from torchvision import transforms

# Taille d'entrée du réseau (carré). Toutes les images y sont ramenées.
IMAGE_SIZE: int = 128

# Moyennes/écarts-types standard d'ImageNet, par canal RGB. Convention
# universelle : centre les pixels pour faciliter la convergence.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_train_transform() -> transforms.Compose:
    """Pipeline d'entraînement : resize + augmentation + normalisation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def build_val_transform() -> transforms.Compose:
    """Pipeline de validation : resize + normalisation, SANS augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])