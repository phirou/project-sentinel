"""
Définition du réseau de neurones convolutif (CNN) pour la classification
binaire humain / non-humain de Sentinel.

Architecture : 3 blocs convolutifs (conv → ReLU → max-pool) qui extraient
des motifs de plus en plus abstraits, suivis de couches entièrement
connectées qui prennent la décision finale.

Le réseau s'adapte automatiquement à la taille d'entrée (IMAGE_SIZE) grâce
à un pooling adaptatif avant les couches denses : changer IMAGE_SIZE dans
transforms.py ne casse donc PAS le modèle.

Le nombre de classes est paramétrable (num_classes) : passer de binaire
(2) à multi-classes (3+) ne demandera qu'un changement d'argument.

Auteur : Léo
Projet : Sentinel — classification de cibles radar
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SentinelCNN(nn.Module):
    """CNN simple pour classification d'images, écrit de zéro."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        # ─── Extraction de features (3 blocs convolutifs) ───────────────
        # Chaque bloc : convolution → activation ReLU → max-pooling /2.
        self.features = nn.Sequential(
            # Bloc 1 : 3 canaux (RGB) → 16 cartes de features.
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # taille / 2

            # Bloc 2 : 16 → 32 cartes.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloc 3 : 32 → 64 cartes.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Pooling adaptatif : ramène n'importe quelle taille spatiale à 4×4.
        # C'est lui qui rend le réseau indépendant de IMAGE_SIZE.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ─── Classifieur (couches entièrement connectées) ───────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # aplatit 64×4×4 en un vecteur de 1024
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),               # désactive 50% des neurones (anti-overfit)
            nn.Linear(128, num_classes),   # décision finale : 1 score par classe
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : image → scores par classe."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test avec un tenseur factice (4 images RGB de 128×128) pour vérifier
    # que les dimensions s'enchaînent sans erreur, SANS aucune vraie image.
    model = SentinelCNN(num_classes=2)
    fake_batch = torch.randn(4, 3, 128, 128)   # (batch, canaux, H, W)
    output = model(fake_batch)
    print(f"Entrée  : {fake_batch.shape}")
    print(f"Sortie  : {output.shape}")          # attendu : (4, 2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {n_params:,}")