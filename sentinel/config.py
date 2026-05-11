"""
Configuration centralisée de Sentinel via YAML.

Ce module charge un fichier YAML, le valide avec Pydantic, et expose un
objet typé utilisable dans toute l'application. Cela évite les valeurs
hardcodées éparpillées dans le code et permet de tuner Sentinel sans
modifier une ligne de Python.

Hiérarchie de chargement (du plus général au plus spécifique) :
    1. config/default.yaml       (toujours chargé, commité dans Git)
    2. config/local.yaml         (overrides développement, gitignored)
    3. fichier --config <path>   (override total, ligne de commande)

Validation Pydantic au chargement : si la config est invalide,
Sentinel refuse de démarrer avec un message d'erreur explicite.
C'est le pattern "fail fast" : on plante au boot plutôt qu'au runtime.

Auteur : Léo
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class RadarSourceType(str, Enum):
    """Type de source radar utilisable par Sentinel."""

    SIMULATOR = "simulator"  # simulateur Python, pas de hardware requis
    UART = "uart"            # vrai LD2450 branché en UART


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLES DE CONFIG (sections du YAML)
# ─────────────────────────────────────────────────────────────────────────────

class RadarConfig(BaseModel):
    """Configuration de la source radar."""

    source: RadarSourceType = Field(
        default=RadarSourceType.SIMULATOR,
        description="Type de source : 'simulator' ou 'uart'",
    )

    # Paramètres UART (utilisés seulement si source = 'uart')
    uart_port: str = Field(
        default="/dev/ttyAMA0",
        description="Chemin du port série (Pi: /dev/ttyAMA0 ou /dev/ttyUSB0)",
    )
    uart_baudrate: int = Field(
        default=256000,
        description="Vitesse UART du LD2450 (toujours 256000 par défaut)",
    )

    # Paramètres simulateur (utilisés seulement si source = 'simulator')
    simulator_targets: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Nombre de cibles aléatoires dans le simulateur (0 à 3)",
    )


class WebConfig(BaseModel):
    """Configuration du serveur web."""

    host: str = Field(default="0.0.0.0", description="Interface d'écoute")
    port: int = Field(default=8000, ge=1, le=65535)
    enable_dashboard: bool = Field(
        default=True,
        description="Servir le dashboard HTML statique",
    )


class PersistenceConfig(BaseModel):
    """Configuration de la persistance SQLite."""

    enabled: bool = Field(default=True)
    db_path: str = Field(
        default="data/sentinel.db",
        description="Chemin du fichier SQLite",
    )


class LoggingConfig(BaseModel):
    """Configuration du logging Python."""

    level: str = Field(
        default="INFO",
        description="Niveau global : DEBUG, INFO, WARNING, ERROR",
    )
    stats_interval_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Fréquence d'affichage des stats agrégées",
    )


class SentinelConfig(BaseModel):
    """
    Configuration racine de Sentinel, agrégeant toutes les sections.

    C'est l'objet retourné par load_config(). Tout le code de Sentinel
    consomme cet objet plutôt que des constantes globales, ce qui permet
    de faire des tests avec différentes configs sans toucher au reste.
    """

    radar: RadarConfig = Field(default_factory=RadarConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    """Charge un fichier YAML en dict, ou retourne {} si le fichier n'existe pas."""
    if not path.exists():
        logger.debug("Fichier de config absent : %s", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Le fichier {path} doit contenir un dictionnaire YAML à la racine"
        )

    logger.info("Config chargée depuis : %s", path)
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Fusion récursive de deux dicts, override écrase base.

    Permet d'avoir un default.yaml complet et un local.yaml qui ne
    contient que les valeurs à modifier (sans devoir tout recopier).
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_file: Optional[Path] = None) -> SentinelConfig:
    """
    Charge et valide la configuration Sentinel.

    Si config_file est fourni, ne charge QUE ce fichier (mode production
    avec un YAML complet et autonome). Sinon, charge default.yaml puis
    fusionne avec local.yaml si présent (mode développement).

    Args:
        config_file: chemin vers un YAML autonome, ou None pour le mode
                     hiérarchique default.yaml + local.yaml.

    Returns:
        Une SentinelConfig validée.

    Raises:
        pydantic.ValidationError: si la config est mal formée.
    """
    # Racine du projet : 2 dossiers au-dessus de ce fichier (sentinel/config.py).
    project_root = Path(__file__).parent.parent

    if config_file is not None:
        # Mode "config explicite" : on charge UN seul fichier.
        raw_config = _load_yaml(Path(config_file))
    else:
        # Mode hiérarchique : default.yaml + local.yaml en override.
        default_path = project_root / "config" / "default.yaml"
        local_path = project_root / "config" / "local.yaml"

        defaults = _load_yaml(default_path)
        overrides = _load_yaml(local_path)
        raw_config = _deep_merge(defaults, overrides)

    # Validation Pydantic : transforme le dict en objet typé,
    # et lève une erreur claire si quelque chose cloche.
    return SentinelConfig(**raw_config)