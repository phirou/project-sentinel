"""Benchmark comparatif des modèles YOLO du projet Sentinel.

Compare objectivement plusieurs modèles YOLO (ultralytics) sur l'ensemble des
vidéos de test, à plusieurs résolutions d'inférence, afin de choisir le modèle
le mieux adapté à un futur déploiement sur Raspberry Pi 5 (CPU, sans NPU).

Pour chaque combinaison (modèle × vidéo × résolution) le script parcourt la
vidéo et mesure : FPS d'inférence (hors décodage), nombre total de détections,
répartition et confiance par classe, sensibilité aux personnes (frames avec au
moins une personne, nombre moyen de personnes par frame).

Sorties :
    - un tableau récapitulatif dans le terminal ;
    - tools/benchmark/results.json  (résultats structurés, réexploitables) ;
    - tools/benchmark/results.md    (tableaux lisibles, pour référence) ;
    - tools/benchmark/samples/      (1 frame annotée par modèle × vidéo).

    ┌──────────────────────────────────────────────────────────────────────┐
    │ IMPORTANT — ce benchmark tourne sur MPS (Mac), la cible est un Pi 5   │
    │ en CPU. Les FPS ABSOLUS ne sont donc PAS transférables au Pi. Ce qui  │
    │ se transfère : le CLASSEMENT relatif des modèles (n < s en vitesse,   │
    │ etc.) et TOUTES les métriques de détection (sensibilité, confiance,   │
    │ nombre de détections), qui ne dépendent pas du matériel.              │
    │ Pour des FPS au caractère « CPU », relancer avec --device cpu.        │
    └──────────────────────────────────────────────────────────────────────┘

Lancement — depuis la racine du repo, venv activé ::

    python tools/benchmark/benchmark_models.py

Options utiles (les valeurs par défaut sont les constantes ci-dessous) ::

    python tools/benchmark/benchmark_models.py --device cpu   # FPS type CPU
    python tools/benchmark/benchmark_models.py --stride 5     # + rapide
    python tools/benchmark/benchmark_models.py --limit 60     # test éclair
    python tools/benchmark/benchmark_models.py --no-samples   # sans vignettes
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRA_LOGGER

# On réduit le bavardage d'ultralytics : seuls nos propres logs structurent la sortie.
import logging

ULTRA_LOGGER.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration (valeurs par défaut — surchargées par les options CLI)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DOSSIER_BENCH = Path(__file__).resolve().parent
DOSSIER_VIDEOS = REPO_ROOT / "tools" / "detection_viewer" / "videos_test"
DOSSIER_SAMPLES = DOSSIER_BENCH / "samples"

# Modèles comparés (poids présents à la racine du repo).
MODELES = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]

# Résolutions d'inférence testées (paramètre imgsz d'ultralytics).
RESOLUTIONS = [640, 1280]

# Échantillonnage temporel : on analyse 1 frame sur FRAME_STRIDE. Comme le
# MÊME stride s'applique à tous les modèles / toutes les résolutions, la
# comparaison reste parfaitement équitable : on compare exactement les mêmes
# frames dans tous les cas. Réduire le stride = plus de précision, plus lent.
FRAME_STRIDE = 3

# Nombre maximal de frames RÉELLEMENT analysées par vidéo (0 = toute la vidéo).
# Utile pour un test éclair ; en run normal on laisse 0 pour tout parcourir.
MAX_FRAMES_PAR_VIDEO = 0

# Seuil de confiance des détections (constant pour tous → comparaison équitable).
CONF = 0.25

# Résolution à partir de laquelle on sauvegarde les frames annotées d'exemple.
SAMPLE_IMGSZ = 1280

# Extensions vidéo reconnues dans le dossier de test.
EXTENSIONS_VIDEO = {".mp4", ".mov", ".avi", ".mkv"}


def detecter_device() -> str:
    """Choisit le meilleur device disponible : MPS (Apple), CUDA, sinon CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchroniser(device: str) -> None:
    """Force l'achèvement des calculs GPU avant de mesurer le temps.

    MPS et CUDA exécutent les calculs de façon asynchrone : sans cette
    synchronisation, le chronomètre s'arrêterait avant la fin réelle de
    l'inférence et les FPS seraient artificiellement gonflés.
    """
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Cœur du benchmark : analyse d'une vidéo par un modèle à une résolution
# ---------------------------------------------------------------------------


def analyser_video(modele: YOLO, chemin: Path, imgsz: int, device: str,
                   stride: int, max_frames: int, capturer_sample: bool) -> dict:
    """Parcourt une vidéo et renvoie les métriques pour (ce modèle, cette résolution).

    Le temps mesuré est celui de la seule inférence (predict), décodage vidéo
    EXCLU : on décode d'abord la frame, puis on chronomètre uniquement l'appel
    au modèle, avec synchronisation du device.

    Lève une exception si la vidéo est illisible — l'appelant l'attrape pour
    ne pas interrompre tout le run.
    """
    cap = cv2.VideoCapture(str(chemin))
    if not cap.isOpened():
        raise RuntimeError(f"vidéo illisible : {chemin.name}")

    # Accumulateurs bruts (permettent une agrégation exacte a posteriori).
    frames_traitees = 0
    temps_inference_s = 0.0                         # temps total d'inférence (hors décodage)
    somme_speed = defaultdict(float)                # détail preprocess/inference/postprocess (ms)
    detections_total = 0
    somme_confiance = 0.0
    compte_classe = defaultdict(int)                # classe → nb de détections
    somme_conf_classe = defaultdict(float)          # classe → somme des confiances
    frames_avec_personne = 0
    personnes_total = 0

    # Pour la frame annotée d'exemple : on retient la frame la plus « riche »
    # (celle qui a le plus de détections), plus représentative qu'une frame prise au hasard.
    meilleur_nb_det = -1
    meilleure_annotation = None

    index = 0
    while True:
        # grab() avance sans décoder (rapide) ; on ne décode (retrieve) que les
        # frames réellement analysées → on ne paie pas le décodage 4K des frames sautées.
        if not cap.grab():
            break
        if index % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break

            # --- Inférence chronométrée (décodage déjà fait, donc exclu) -----
            t0 = time.perf_counter()
            resultat = modele.predict(frame, imgsz=imgsz, device=device, conf=CONF, verbose=False)[0]
            synchroniser(device)
            temps_inference_s += time.perf_counter() - t0

            for cle, valeur in resultat.speed.items():  # preprocess / inference / postprocess (ms)
                somme_speed[cle] += valeur

            # --- Comptage des détections de la frame ------------------------
            classes = resultat.boxes.cls.tolist()
            confiances = resultat.boxes.conf.tolist()
            nb_personnes_frame = 0
            for cls_id, conf in zip(classes, confiances):
                nom = resultat.names[int(cls_id)]
                detections_total += 1
                somme_confiance += conf
                compte_classe[nom] += 1
                somme_conf_classe[nom] += conf
                if nom == "person":
                    nb_personnes_frame += 1

            if nb_personnes_frame > 0:
                frames_avec_personne += 1
                personnes_total += nb_personnes_frame

            # --- Frame annotée d'exemple (uniquement à la résolution dédiée) -
            if capturer_sample and len(classes) > meilleur_nb_det:
                meilleur_nb_det = len(classes)
                meilleure_annotation = resultat.plot()  # image BGR annotée par ultralytics

            frames_traitees += 1
            if max_frames and frames_traitees >= max_frames:
                break
        index += 1

    cap.release()

    if frames_traitees == 0:
        raise RuntimeError(f"aucune frame exploitable : {chemin.name}")

    # --- Consolidation des métriques -------------------------------------
    fps_detection = frames_traitees / temps_inference_s if temps_inference_s > 0 else 0.0
    conf_moyenne = somme_confiance / detections_total if detections_total else 0.0
    conf_par_classe = {
        classe: round(somme_conf_classe[classe] / compte_classe[classe], 3)
        for classe in compte_classe
    }
    speed_moyen = {cle: round(valeur / frames_traitees, 2) for cle, valeur in somme_speed.items()}

    return {
        "modele": modele.ckpt_path.split("/")[-1] if hasattr(modele, "ckpt_path") else "?",
        "video": chemin.name,
        "imgsz": imgsz,
        "frames_traitees": frames_traitees,
        "fps_detection": round(fps_detection, 2),
        "ms_par_frame": speed_moyen,
        "detections_total": detections_total,
        "detections_par_classe": dict(sorted(compte_classe.items(), key=lambda kv: -kv[1])),
        "confiance_moyenne": round(conf_moyenne, 3),
        "confiance_par_classe": conf_par_classe,
        "frames_avec_personne": frames_avec_personne,
        "pct_frames_avec_personne": round(100 * frames_avec_personne / frames_traitees, 1),
        "personnes_par_frame": round(personnes_total / frames_traitees, 2),
        # Champs bruts conservés pour une agrégation exacte (voir agreger()).
        "_brut": {
            "temps_inference_s": temps_inference_s,
            "somme_confiance": somme_confiance,
            "personnes_total": personnes_total,
        },
        "erreur": None,
        # Frame annotée éventuelle (retirée avant sérialisation JSON).
        "_annotation": meilleure_annotation,
    }


# ---------------------------------------------------------------------------
# Agrégation et mise en forme
# ---------------------------------------------------------------------------


def agreger(resultats: list[dict]) -> list[dict]:
    """Agrège les résultats par (modèle, imgsz) en cumulant toutes les vidéos.

    L'agrégation part des sommes brutes (frames, temps, détections…) pour que
    les taux (FPS, %, moyennes) soient exacts et non des « moyennes de moyennes ».
    """
    cumuls: dict[tuple, dict] = {}
    for r in resultats:
        if r["erreur"]:
            continue
        cle = (r["modele"], r["imgsz"])
        c = cumuls.setdefault(cle, {
            "modele": r["modele"], "imgsz": r["imgsz"], "videos": 0,
            "frames": 0, "temps_s": 0.0, "detections": 0, "somme_conf": 0.0,
            "frames_personne": 0, "personnes": 0,
        })
        c["videos"] += 1
        c["frames"] += r["frames_traitees"]
        c["temps_s"] += r["_brut"]["temps_inference_s"]
        c["detections"] += r["detections_total"]
        c["somme_conf"] += r["_brut"]["somme_confiance"]
        c["frames_personne"] += r["frames_avec_personne"]
        c["personnes"] += r["_brut"]["personnes_total"]

    agrege = []
    for c in cumuls.values():
        frames = c["frames"] or 1
        agrege.append({
            "modele": c["modele"],
            "imgsz": c["imgsz"],
            "videos": c["videos"],
            "frames_traitees": c["frames"],
            "fps_detection": round(c["frames"] / c["temps_s"], 2) if c["temps_s"] else 0.0,
            "detections_total": c["detections"],
            "confiance_moyenne": round(c["somme_conf"] / c["detections"], 3) if c["detections"] else 0.0,
            "pct_frames_avec_personne": round(100 * c["frames_personne"] / frames, 1),
            "personnes_par_frame": round(c["personnes"] / frames, 2),
        })
    # Tri lisible : par résolution, puis modèle.
    return sorted(agrege, key=lambda a: (a["imgsz"], a["modele"]))


def _tableau_terminal(entetes: list[str], lignes: list[list], aligns: str) -> str:
    """Construit un tableau texte lisible (colonnes alignées, en-tête souligné)."""
    largeurs = [len(e) for e in entetes]
    for ligne in lignes:
        for i, cell in enumerate(ligne):
            largeurs[i] = max(largeurs[i], len(str(cell)))

    def formater(cells):
        morceaux = []
        for i, cell in enumerate(cells):
            texte = str(cell)
            morceaux.append(texte.rjust(largeurs[i]) if aligns[i] == "r" else texte.ljust(largeurs[i]))
        return "  ".join(morceaux)

    lignes_txt = [formater(entetes), "  ".join("─" * l for l in largeurs)]
    lignes_txt += [formater(ligne) for ligne in lignes]
    return "\n".join(lignes_txt)


def _tableau_markdown(entetes: list[str], lignes: list[list], aligns: str) -> str:
    """Construit un tableau Markdown (avec alignement des colonnes)."""
    sep = {"r": "---:", "l": ":---"}
    out = ["| " + " | ".join(entetes) + " |",
           "| " + " | ".join(sep[a] for a in aligns) + " |"]
    for ligne in lignes:
        out.append("| " + " | ".join(str(c) for c in ligne) + " |")
    return "\n".join(out)


def _court(nom_video: str) -> str:
    """Nom de vidéo raccourci pour les tableaux (sans extension)."""
    return Path(nom_video).stem


def afficher_tableaux(resultats: list[dict], agrege: list[dict], device: str, stride: int) -> None:
    """Affiche dans le terminal le récapitulatif détaillé puis agrégé."""
    print("\n" + "═" * 78)
    print("  RÉSUMÉ PAR MODÈLE ET RÉSOLUTION  (agrégé sur toutes les vidéos)")
    print("═" * 78)
    entetes = ["Modèle", "imgsz", "FPS", "Dét.tot", "Pers/f", "%f pers", "Conf.moy"]
    lignes = [[a["modele"], a["imgsz"], f"{a['fps_detection']:.1f}", a["detections_total"],
               f"{a['personnes_par_frame']:.2f}", f"{a['pct_frames_avec_personne']:.1f}",
               f"{a['confiance_moyenne']:.3f}"] for a in agrege]
    print(_tableau_terminal(entetes, lignes, "lrrrrrr"))

    print("\n" + "═" * 78)
    print("  DÉTAIL PAR VIDÉO")
    print("═" * 78)
    ok = [r for r in resultats if not r["erreur"]]
    ok.sort(key=lambda r: (r["modele"], r["imgsz"], r["video"]))
    entetes = ["Modèle", "imgsz", "Vidéo", "FPS", "Dét.", "Pers/f", "%f pers", "Conf."]
    lignes = [[r["modele"], r["imgsz"], _court(r["video"]), f"{r['fps_detection']:.1f}",
               r["detections_total"], f"{r['personnes_par_frame']:.2f}",
               f"{r['pct_frames_avec_personne']:.1f}", f"{r['confiance_moyenne']:.3f}"] for r in ok]
    print(_tableau_terminal(entetes, lignes, "lrlrrrrr"))

    erreurs = [r for r in resultats if r["erreur"]]
    if erreurs:
        print("\n⚠ Combinaisons en erreur (ignorées) :")
        for r in erreurs:
            print(f"   - {r['modele']} × {r['video']} @ {r['imgsz']} : {r['erreur']}")

    print("\n" + "─" * 78)
    print(f"  Device : {device.upper()}   |   stride : 1/{stride}   |   conf : {CONF}")
    if device != "cpu":
        print("  ⚠ FPS mesurés sur", device.upper(), "→ NON transférables au Pi 5 (CPU).")
        print("    Transférables : le classement relatif des modèles et toutes les")
        print("    métriques de détection. Pour des FPS type CPU : --device cpu.")
    print("─" * 78 + "\n")


# ---------------------------------------------------------------------------
# Sauvegardes JSON / Markdown
# ---------------------------------------------------------------------------


def sauver_json(resultats: list[dict], agrege: list[dict], meta: dict) -> Path:
    """Écrit les résultats structurés dans results.json (champs internes retirés)."""
    propres = []
    for r in resultats:
        r2 = {k: v for k, v in r.items() if not k.startswith("_")}
        propres.append(r2)
    chemin = DOSSIER_BENCH / "results.json"
    chemin.write_text(
        json.dumps({"meta": meta, "resume": agrege, "detail": propres}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return chemin


def sauver_markdown(resultats: list[dict], agrege: list[dict], meta: dict) -> Path:
    """Écrit un rapport lisible dans results.md (pour référence / portfolio)."""
    lignes = ["# Benchmark des modèles YOLO — Project Sentinel", ""]
    lignes.append(f"*Généré le {meta['date']} — device : **{meta['device'].upper()}**, "
                  f"FRAME_STRIDE : {meta['frame_stride']}, conf : {meta['conf']}.*")
    lignes += ["",
               "> ⚠️ **Cible réelle : Raspberry Pi 5 en CPU.** Ce benchmark tourne sur "
               f"`{meta['device'].upper()}`. Les **FPS absolus ne sont pas transférables** au Pi ; "
               "en revanche le **classement relatif** des modèles et **toutes les métriques de "
               "détection** (sensibilité, confiance, nombre de détections) le sont, car elles ne "
               "dépendent pas du matériel. Pour des FPS au caractère CPU : `--device cpu`.", ""]

    # Résumé agrégé.
    lignes += ["## Résumé par modèle et résolution", "",
               "Agrégé sur toutes les vidéos. `%f pers` = part des frames contenant au moins une "
               "personne (proxy de sensibilité) ; `Pers/f` = nombre moyen de personnes par frame.", ""]
    entetes = ["Modèle", "imgsz", "FPS", "Dét. tot.", "Pers/f", "%f pers", "Conf. moy."]
    lignes_tab = [[a["modele"], a["imgsz"], f"{a['fps_detection']:.1f}", a["detections_total"],
                   f"{a['personnes_par_frame']:.2f}", f"{a['pct_frames_avec_personne']:.1f}",
                   f"{a['confiance_moyenne']:.3f}"] for a in agrege]
    lignes.append(_tableau_markdown(entetes, lignes_tab, "lrrrrrr"))

    # Détail par vidéo.
    lignes += ["", "## Détail par vidéo", ""]
    ok = sorted((r for r in resultats if not r["erreur"]), key=lambda r: (r["modele"], r["imgsz"], r["video"]))
    entetes = ["Modèle", "imgsz", "Vidéo", "FPS", "Dét.", "Pers/f", "%f pers", "Conf."]
    lignes_tab = [[r["modele"], r["imgsz"], _court(r["video"]), f"{r['fps_detection']:.1f}",
                   r["detections_total"], f"{r['personnes_par_frame']:.2f}",
                   f"{r['pct_frames_avec_personne']:.1f}", f"{r['confiance_moyenne']:.3f}"] for r in ok]
    lignes.append(_tableau_markdown(entetes, lignes_tab, "lrlrrrrr"))

    # Répartition par classe (agrégée par modèle × imgsz).
    lignes += ["", "## Répartition des détections par classe", "",
               "Agrégée sur toutes les vidéos, confiance moyenne entre parenthèses.", ""]
    classes_par_cle: dict[tuple, dict] = {}
    for r in ok:
        cle = (r["modele"], r["imgsz"])
        acc = classes_par_cle.setdefault(cle, {})
        for classe, nb in r["detections_par_classe"].items():
            a = acc.setdefault(classe, {"nb": 0, "somme_conf": 0.0})
            a["nb"] += nb
            a["somme_conf"] += r["confiance_par_classe"].get(classe, 0.0) * nb
    entetes = ["Modèle", "imgsz", "Classes détectées (nb, conf. moy.)"]
    lignes_tab = []
    for (modele, imgsz), acc in sorted(classes_par_cle.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        parts = [f"{classe} : {a['nb']} ({a['somme_conf'] / a['nb']:.2f})"
                 for classe, a in sorted(acc.items(), key=lambda kv: -kv[1]["nb"])]
        lignes_tab.append([modele, imgsz, " · ".join(parts)])
    lignes.append(_tableau_markdown(entetes, lignes_tab, "lrl"))

    lignes += ["", "## Méthodologie", "",
               f"- **Modèles** : {', '.join(meta['modeles'])}.",
               f"- **Résolutions** (`imgsz`) : {', '.join(map(str, meta['resolutions']))}.",
               f"- **Échantillonnage** : 1 frame sur {meta['frame_stride']} "
               "(le même pour tous → comparaison équitable, exactement les mêmes frames comparées).",
               f"- **Seuil de confiance** : {meta['conf']} (constant).",
               "- **FPS** : temps de la seule inférence `predict()` (décodage vidéo exclu), "
               "avec synchronisation du device pour une mesure fiable.",
               "- Une frame annotée par modèle × vidéo est disponible dans `samples/`.", ""]

    chemin = DOSSIER_BENCH / "results.md"
    chemin.write_text("\n".join(lignes), encoding="utf-8")
    return chemin


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def lister_videos() -> list[Path]:
    """Liste triée des vidéos de test exploitables."""
    if not DOSSIER_VIDEOS.is_dir():
        return []
    return sorted(f for f in DOSSIER_VIDEOS.iterdir() if f.suffix.lower() in EXTENSIONS_VIDEO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark des modèles YOLO (Project Sentinel).")
    parser.add_argument("--device", default=None, help="mps | cuda | cpu (défaut : auto-détection)")
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE, help="analyser 1 frame sur N")
    parser.add_argument("--limit", type=int, default=MAX_FRAMES_PAR_VIDEO, help="max frames/vidéo (0 = tout)")
    parser.add_argument("--no-samples", action="store_true", help="ne pas sauvegarder de frames annotées")
    args = parser.parse_args()

    device = args.device or detecter_device()
    videos = lister_videos()
    if not videos:
        print(f"Aucune vidéo trouvée dans {DOSSIER_VIDEOS}")
        return

    print("═" * 78)
    print("  BENCHMARK YOLO — PROJECT SENTINEL")
    print("═" * 78)
    print(f"  Device       : {device.upper()}")
    print(f"  Modèles      : {', '.join(MODELES)}")
    print(f"  Résolutions  : {', '.join(map(str, RESOLUTIONS))}")
    print(f"  Vidéos       : {len(videos)}  ({DOSSIER_VIDEOS.relative_to(REPO_ROOT)})")
    print(f"  Stride       : 1 frame sur {args.stride}" + (f"  (max {args.limit}/vidéo)" if args.limit else ""))
    total_combi = len(MODELES) * len(videos) * len(RESOLUTIONS)
    print(f"  Combinaisons : {total_combi}  (modèle × vidéo × résolution)")
    print("═" * 78 + "\n")

    if not args.no_samples:
        DOSSIER_SAMPLES.mkdir(exist_ok=True)

    resultats: list[dict] = []
    debut_total = time.perf_counter()
    combi = 0

    for nom_modele in MODELES:
        chemin_poids = REPO_ROOT / nom_modele
        try:
            modele = YOLO(str(chemin_poids))
        except Exception as exc:
            print(f"✗ Modèle « {nom_modele} » non chargeable : {exc} — ignoré.")
            # On enregistre l'échec pour chaque combinaison concernée.
            for video in videos:
                for imgsz in RESOLUTIONS:
                    resultats.append({"modele": nom_modele, "video": video.name, "imgsz": imgsz,
                                      "erreur": f"modèle non chargeable : {exc}"})
            continue

        for imgsz in RESOLUTIONS:
            # Préchauffage : la 1re inférence à une résolution donnée compile les
            # noyaux (surtout sur MPS) et serait anormalement lente si chronométrée.
            modele.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                           imgsz=imgsz, device=device, verbose=False)
            synchroniser(device)

            for video in videos:
                combi += 1
                capturer = (not args.no_samples) and imgsz == SAMPLE_IMGSZ
                prefixe = f"[{combi:2d}/{total_combi}] {nom_modele:12s} {imgsz:5d}  {_court(video.name):32s}"
                try:
                    r = analyser_video(modele, video, imgsz, device, args.stride, args.limit, capturer)
                    r["modele"] = nom_modele  # nom sûr et cohérent
                    print(f"{prefixe} → {r['fps_detection']:6.1f} FPS  "
                          f"{r['detections_total']:5d} dét.  {r['pct_frames_avec_personne']:5.1f}% pers.")

                    # Sauvegarde de la frame annotée d'exemple, si capturée.
                    annotation = r.pop("_annotation", None)
                    if annotation is not None:
                        nom = f"{Path(nom_modele).stem}__{_court(video.name)}.jpg"
                        cv2.imwrite(str(DOSSIER_SAMPLES / nom), annotation)
                    resultats.append(r)
                except Exception as exc:
                    print(f"{prefixe} → ✗ ERREUR : {exc}")
                    resultats.append({"modele": nom_modele, "video": video.name, "imgsz": imgsz,
                                      "erreur": str(exc)})

    duree = time.perf_counter() - debut_total

    # --- Agrégation, affichage et sauvegardes ----------------------------
    agrege = agreger(resultats)
    afficher_tableaux(resultats, agrege, device, args.stride)

    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "frame_stride": args.stride,
        "max_frames_par_video": args.limit,
        "conf": CONF,
        "resolutions": RESOLUTIONS,
        "modeles": MODELES,
        "duree_totale_s": round(duree, 1),
        "note_cible": "Benchmark sur " + device.upper() + " ; cible réelle : Raspberry Pi 5 CPU. "
                      "FPS absolus non transférables ; classement relatif et métriques de détection transférables.",
    }
    chemin_json = sauver_json(resultats, agrege, meta)
    chemin_md = sauver_markdown(resultats, agrege, meta)

    print(f"Durée totale : {duree:.1f} s")
    print(f"Résultats    : {chemin_json.relative_to(REPO_ROOT)}")
    print(f"               {chemin_md.relative_to(REPO_ROOT)}")
    if not args.no_samples:
        print(f"Vignettes    : {DOSSIER_SAMPLES.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
