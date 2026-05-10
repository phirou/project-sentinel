"""
Modèles de données du domaine Sentinel.

Ce module définit les structures de données fondamentales utilisées dans
toute l'application : représentation d'une cible radar, d'une trame complète,
et des enums associés.

Ces modèles sont basés sur Pydantic v2, ce qui apporte :
- Validation automatique des types et des contraintes
- Sérialisation JSON immédiate (utile pour l'API et le WebSocket)
- Auto-documentation via les schémas générés

Auteur : Léo R
Projet : Sentinel — Système de surveillance autonome
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DU CAPTEUR LD2450
# ─────────────────────────────────────────────────────────────────────────────

# Le LD2450 émet à 24 GHz et peut suivre jusqu'à 3 cibles simultanément.
# Au-delà, le firmware Hi-Link choisit lesquelles ignorer (logique propriétaire).
MAX_TARGETS_PER_FRAME: int = 3

# Champ de vue azimutal du capteur en degrés (±60° de chaque côté de l'axe Y).
RADAR_FOV_DEGREES: float = 120.0

# Portée maximale annoncée par le constructeur sur cible humaine, en mètres.
RADAR_MAX_RANGE_METERS: float = 8.0


# ─────────────────────────────────────────────────────────────────────────────
# ENUMÉRATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TargetState(str, Enum):
    """
    État cinématique d'une cible, dérivé du signe de sa vitesse radiale.

    Le LD2450 mesure la vitesse par effet Doppler. Une vitesse positive
    signifie que la cible s'éloigne du radar, négative qu'elle s'en approche.
    """

    APPROACHING = "approaching"  # vitesse < 0 : la cible se rapproche
    RECEDING = "receding"        # vitesse > 0 : la cible s'éloigne
    STATIONARY = "stationary"    # vitesse ≈ 0 : la cible est immobile


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLES DE DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

class Target(BaseModel):
    """
    Représente une cible détectée par le radar à un instant donné.

    Les coordonnées (x, y) sont en millimètres dans le repère du radar :
        - Y positif = devant le radar
        - X positif = à droite, X négatif = à gauche
        - L'origine (0, 0) est la position du capteur

    La vitesse est en cm/s (convention Hi-Link), positive si la cible
    s'éloigne, négative si elle s'approche.
    """

    # Identifiant du slot dans la trame (0, 1 ou 2). Ce n'est pas un ID
    # de tracking persistant — pour ça il faudra un TargetManager dédié
    # qui assigne des IDs stables entre les trames.
    slot: int = Field(..., ge=0, le=2, description="Index de la cible dans la trame (0..2)")

    # Position cartésienne dans le repère radar.
    x_mm: int = Field(..., description="Coordonnée X en millimètres (positif = droite)")
    y_mm: int = Field(..., description="Coordonnée Y en millimètres (positif = devant)")

    # Vitesse radiale (mesurée par effet Doppler).
    speed_cms: int = Field(..., description="Vitesse en cm/s (>0 s'éloigne, <0 s'approche)")

    # Résolution de la mesure radar : précision de localisation en mm.
    # Utile pour pondérer la confiance d'une détection.
    resolution_mm: int = Field(..., ge=0, description="Résolution de la mesure en mm")

    # ─── Propriétés calculées ───────────────────────────────────────────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def distance_mm(self) -> float:
        """Distance euclidienne entre le radar et la cible (en mm)."""
        return (self.x_mm ** 2 + self.y_mm ** 2) ** 0.5

    @computed_field  # type: ignore[prop-decorator]
    @property
    def distance_m(self) -> float:
        """Distance en mètres, plus pratique pour l'affichage."""
        return self.distance_mm / 1000.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def angle_degrees(self) -> float:
        """
        Angle azimutal de la cible par rapport à l'axe Y du radar.

        Convention :
            - 0° = pile devant le radar
            - +60° = bord droit du champ de vue
            - -60° = bord gauche du champ de vue
        """
        from math import atan2, degrees
        # atan2(x, y) donne l'angle par rapport à l'axe Y (et non X), ce qui
        # correspond à notre convention "0° = devant".
        return degrees(atan2(self.x_mm, self.y_mm))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def speed_ms(self) -> float:
        """Vitesse en m/s, plus pratique pour l'affichage."""
        return self.speed_cms / 100.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def state(self) -> TargetState:
        """État cinématique de la cible (approche, éloignement, stationnaire)."""
        # Un seuil de 5 cm/s évite que le bruit Doppler ne fasse osciller
        # une cible immobile entre les états approche/éloignement.
        if self.speed_cms < -5:
            return TargetState.APPROACHING
        elif self.speed_cms > 5:
            return TargetState.RECEDING
        return TargetState.STATIONARY


class RadarFrame(BaseModel):
    """
    Représente une trame complète issue du LD2450 à un instant T.

    Une trame contient jusqu'à 3 cibles. Les slots vides du capteur
    (cibles inexistantes) ne sont PAS inclus dans la liste targets :
    le filtrage est fait par le parser. Ainsi, len(targets) reflète
    réellement le nombre de cibles détectées.
    """

    # Horodatage de la réception de la trame, en UTC.
    # Important d'utiliser UTC pour éviter les bugs de fuseau horaire.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Liste des cibles détectées (0 à 3 éléments).
    targets: list[Target] = Field(default_factory=list)

    # Trame brute en hexadécimal, utile pour debug et logs.
    # Optionnel pour ne pas alourdir la mémoire en production.
    raw_hex: Optional[str] = Field(default=None, description="Trame brute en hex (debug)")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_count(self) -> int:
        """Nombre de cibles détectées dans cette trame (0 à 3)."""
        return len(self.targets)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_targets(self) -> bool:
        """True si au moins une cible est présente dans la trame."""
        return self.target_count > 0
