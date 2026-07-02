"""
Comparateur de modèles de détection pour Sentinel.

Fait tourner plusieurs modèles de détection d'objets sur une même vidéo et
compare leurs performances : vitesse (FPS), nombre de détections, répartition
par classe. Produit aussi une vidéo annotée par modèle pour comparaison visuelle.

Objectif : choisir objectivement le meilleur modèle pour le déploiement sur
Raspberry Pi 5 (+ Hailo-8), en arbitrant vitesse vs précision.

Lancement :
    python -m ml.compare_models

Auteur : Léo
Projet : Sentinel — détection temps réel
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


# Modèles à comparer. Les fichiers .pt se téléchargent automatiquement au
# premier chargement (quelques Mo chacun). On commence par 3 modèles Ultralytics.
MODELS_TO_COMPARE: list[str] = [
    "yolov8n.pt",   # nano — le plus rapide
    "yolov8s.pt",   # small — plus précis, plus lent
    "yolo11n.pt",   # génération suivante
]

# Vidéo de test. On utilise un asset d'exemple fourni par Ultralytics.
# Cette URL est reconnue nativement par Ultralytics et téléchargée au besoin.
TEST_VIDEO: str = "test_video.mp4"

# Dossier de sortie pour les vidéos annotées.
OUTPUT_DIR: Path = Path("ml/comparison_output")


def benchmark_model(model_name: str, source: str) -> dict:
    """
    Fait tourner un modèle sur la source (image ou vidéo) et mesure ses perfs.

    Args:
        model_name: nom du modèle (ex: "yolov8n.pt"). Téléchargé si absent.
        source: chemin/URL de l'image ou vidéo à analyser.

    Returns:
        Un dict avec le nom, le FPS moyen, le total de détections et le
        comptage par classe.
    """
    print(f"\n{'='*50}")
    print(f"Modèle : {model_name}")
    print(f"{'='*50}")

    # Chargement du modèle (télécharge les poids au premier appel).
    model = YOLO(model_name)

    # Compteur de détections par classe (nom lisible : "person", "car"...).
    class_counts: Counter = Counter()
    total_detections = 0

    # Chronométrage de l'inférence.
    start = time.perf_counter()

    # stream=True : traite frame par frame sans tout charger en mémoire.
    # verbose=False : coupe les logs internes d'Ultralytics (trop bavards).
    results = model(source, stream=True, verbose=False)

    frame_count = 0
    for result in results:
        frame_count += 1
        # result.boxes contient toutes les détections de la frame.
        for box in result.boxes:
            class_id = int(box.cls)                 # id numérique de la classe
            class_name = model.names[class_id]      # nom lisible correspondant
            class_counts[class_name] += 1
            total_detections += 1

    elapsed = time.perf_counter() - start
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    return {
        "model": model_name,
        "fps": fps,
        "frames": frame_count,
        "total_detections": total_detections,
        "class_counts": dict(class_counts),
    }



def main() -> None:
    print("Comparateur de modèles Sentinel")
    print(f"Source de test : {TEST_VIDEO}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # On benchmark chaque modèle et on stocke les résultats.
    results = []
    for model_name in MODELS_TO_COMPARE:
        stats = benchmark_model(model_name, TEST_VIDEO)
        results.append(stats)

    # --- Tableau récapitulatif ---
    print(f"\n\n{'='*70}")
    print("RÉCAPITULATIF COMPARATIF")
    print(f"{'='*70}")
    print(f"{'Modèle':<15} {'FPS':>8} {'Frames':>8} {'Détections':>12}")
    print(f"{'-'*70}")
    for r in results:
        print(
            f"{r['model']:<15} {r['fps']:>8.1f} "
            f"{r['frames']:>8} {r['total_detections']:>12}"
        )

    # Détail par classe pour chaque modèle.
    print(f"\n{'='*70}")
    print("DÉTECTIONS PAR CLASSE")
    print(f"{'='*70}")
    for r in results:
        print(f"\n{r['model']} :")
        if r["class_counts"]:
            for class_name, count in sorted(
                r["class_counts"].items(), key=lambda x: -x[1]
            ):
                print(f"  {class_name:<20} {count:>5}")
        else:
            print("  (aucune détection)")


if __name__ == "__main__":
    main()
