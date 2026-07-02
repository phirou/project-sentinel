"""Detection Viewer — visualisation temps réel de détections YOLO.

Mini application web locale : un backend FastAPI fait tourner un modèle YOLO
(ultralytics) sur une vidéo lue en boucle, diffuse le flux annoté en MJPEG et
expose des vignettes des objets détectés. Le front (``templates/index.html``)
est une page autonome en HTML/CSS/JS vanilla.

Lancement — depuis la racine du repo, venv activé ::

    uvicorn tools.detection_viewer.app:app --port 8100

puis ouvrir http://localhost:8100
(port 8100 pour ne pas entrer en conflit avec Sentinel, qui écoute sur 8000).

Arrêt : Ctrl+C dans le terminal (deux fois si un flux vidéo est encore ouvert
dans un onglet). Un seul worker (défaut uvicorn) : l'état vit en mémoire.

Endpoints :
    GET  /            page web
    GET  /video_feed  flux MJPEG annoté (multipart/x-mixed-replace)
    GET  /detections  vignettes récentes des objets détectés (JSON + base64)
    GET  /stats       FPS d'inférence, modèle courant, device, nb d'objets
    GET  /sources     sources vidéo disponibles (videos_test/ + caméra)
    POST /set_model   changement de modèle à chaud, ex. {"model": "yolov8s"}
    POST /set_source  changement de source à chaud, ex. {"source": "camera"}
"""

import base64
import itertools
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Racine du repo : les poids .pt et la vidéo de test y sont stockés.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOSSIER_APP = Path(__file__).resolve().parent

# Source vidéo par défaut au démarrage (nom de fichier, cherché d'abord dans
# videos_test/ puis à la racine du repo). L'interface permet ensuite de
# basculer à chaud sur une autre vidéo ou sur la caméra (voir /set_source).
VIDEO_SOURCE = "crowd_street_dense.mp4"

# Vidéos de test proposées dans l'interface. Le dossier est re-scanné à chaque
# appel de /sources : y déposer un fichier suffit, sans redémarrer le serveur.
DOSSIER_VIDEOS = DOSSIER_APP / "videos_test"
EXTENSIONS_VIDEO = {".mp4", ".mov", ".avi", ".mkv"}

# Entrée spéciale « caméra » : la webcam locale pour l'instant, pensée pour
# brancher plus tard la Pi Camera ou un flux RTSP — il suffira de changer
# CIBLE_CAMERA (ex. "rtsp://…").
SOURCE_CAMERA = "camera"
CIBLE_CAMERA: int | str = 0  # index OpenCV de la webcam locale

# Modèles proposés dans l'interface → fichier de poids correspondant.
MODELES = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolo11n": "yolo11n.pt",
}
MODELE_PAR_DEFAUT = "yolov8n"

SEUIL_CONFIANCE = 0.4       # score minimal pour retenir une détection
LARGEUR_MAX_FLUX = 1280     # les frames plus larges sont réduites (bande passante)
QUALITE_JPEG = 80           # qualité d'encodage du flux MJPEG

INTERVALLE_VIGNETTES = 1.0  # au plus une passe d'extraction de crops par seconde
MAX_VIGNETTES = 20          # nombre de vignettes gardées en mémoire
TAILLE_VIGNETTE = 160       # côté maximal (px) d'une vignette

# Couleurs BGR (OpenCV) : vert phosphore #00ff41 pour "person", orange #ff9500
# pour toutes les autres classes — assorties au thème de l'interface.
COULEUR_PERSONNE = (65, 255, 0)
COULEUR_AUTRE = (0, 149, 255)
POLICE = cv2.FONT_HERSHEY_SIMPLEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("detection_viewer")


def detecter_device() -> str:
    """Choisit le meilleur device disponible pour l'inférence."""
    if torch.backends.mps.is_available():  # GPU Apple Silicon
        return "mps"
    if torch.cuda.is_available():  # GPU NVIDIA
        return "cuda"
    return "cpu"


DEVICE = detecter_device()

# ---------------------------------------------------------------------------
# État partagé entre la boucle d'inférence (thread) et les endpoints HTTP
# ---------------------------------------------------------------------------


class EtatPartage:
    """Tout ce qui est produit par le thread d'inférence et lu par l'API."""

    def __init__(self) -> None:
        self.verrou = threading.Lock()
        # Condition notifiée à chaque nouvelle frame : les clients MJPEG
        # attendent dessus au lieu de consommer du CPU en boucle.
        self.frame_prete = threading.Condition(self.verrou)
        self.arret = threading.Event()  # demande d'arrêt propre du thread

        self.frame_jpeg: bytes | None = None  # dernière frame annotée (JPEG)
        self.fps: float = 0.0  # FPS d'inférence lissé (moyenne mobile)
        self.nb_objets: int = 0  # objets détectés sur la dernière frame
        self.modele_actuel: str = MODELE_PAR_DEFAUT
        self.modele_demande: str | None = None  # changement demandé via /set_model
        self.source_actuelle: str = VIDEO_SOURCE  # identifiant de la source en cours
        self.source_demandee: str | None = None  # changement demandé via /set_source
        self.vignettes: deque[dict] = deque(maxlen=MAX_VIGNETTES)  # récentes en tête
        self.ids = itertools.count(1)  # identifiants croissants des vignettes


etat = EtatPartage()

# ---------------------------------------------------------------------------
# Fonctions vision : ouverture de la source, annotation, vignettes
# ---------------------------------------------------------------------------


def lister_sources() -> list[dict]:
    """Construit la liste des sources sélectionnables dans l'interface.

    La vidéo par défaut apparaît en tête, suivie des vidéos du dossier
    videos_test/ (re-scanné à chaque appel), puis de l'entrée « caméra ».
    Chaque source = identifiant (attendu par /set_source) + libellé lisible.
    """
    fichiers = []
    if DOSSIER_VIDEOS.is_dir():
        fichiers = sorted(
            f.name for f in DOSSIER_VIDEOS.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS_VIDEO
        )

    # Compatibilité : la vidéo par défaut peut aussi vivre à la racine du repo.
    if VIDEO_SOURCE not in fichiers and (REPO_ROOT / VIDEO_SOURCE).is_file():
        fichiers.append(VIDEO_SOURCE)

    # La source par défaut passe en tête de liste.
    if VIDEO_SOURCE in fichiers:
        fichiers.remove(VIDEO_SOURCE)
        fichiers.insert(0, VIDEO_SOURCE)

    sources = [
        {
            "id": nom,
            "libelle": Path(nom).stem.replace("_", " ").replace("-", " ")
            + (" (défaut)" if nom == VIDEO_SOURCE else ""),
            "type": "fichier",
        }
        for nom in fichiers
    ]
    # Entrée spéciale, toujours proposée : webcam locale (Pi Camera / RTSP à terme).
    sources.append({"id": SOURCE_CAMERA, "libelle": "caméra (webcam)", "type": "camera"})
    return sources


def resoudre_source(ident: str) -> str | int | None:
    """Traduit un identifiant de source en cible OpenCV (chemin ou index caméra).

    Seuls des identifiants validés par /set_source arrivent ici : on ne résout
    que des noms de fichiers connus, jamais un chemin arbitraire du client.
    """
    if ident == SOURCE_CAMERA:
        return CIBLE_CAMERA
    for dossier in (DOSSIER_VIDEOS, REPO_ROOT):
        chemin = dossier / ident
        if chemin.is_file():
            return str(chemin)
    return None


def ouvrir_source(ident: str) -> tuple[cv2.VideoCapture | None, float]:
    """Ouvre la source identifiée et renvoie (capture, délai cible entre 2 frames)."""
    cible = resoudre_source(ident)
    if cible is None:
        return None, 1 / 30
    cap = cv2.VideoCapture(cible)
    if not cap.isOpened():
        return None, 1 / 30
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Garde-fou : caméras et fichiers exotiques ne renseignent pas toujours leur cadence.
    delai = 1 / fps if 1.0 <= fps <= 240.0 else 1 / 30
    logger.info(
        "Source vidéo ouverte : %s (%.0fx%.0f @ %.0f fps)",
        cible,
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        fps,
    )
    return cap, delai


def charger_modele(nom: str) -> YOLO:
    """Charge un modèle YOLO depuis la racine du repo et le préchauffe."""
    debut = time.perf_counter()
    modele = YOLO(str(REPO_ROOT / MODELES[nom]))
    # Préchauffage : la toute première inférence (compilation des noyaux MPS,
    # allocations…) est lente ; on la paie ici plutôt que sur le flux.
    modele.predict(np.zeros((320, 320, 3), dtype=np.uint8), device=DEVICE, verbose=False)
    logger.info("Modèle « %s » chargé en %.1f s (device : %s)", nom, time.perf_counter() - debut, DEVICE)
    return modele


def reduire_si_besoin(frame: np.ndarray) -> np.ndarray:
    """Plafonne la largeur de la frame à LARGEUR_MAX_FLUX.

    Appliquée AVANT l'inférence : les vidéos 4K du dossier de test restent
    gérables (YOLO, annotation, encodage JPEG et vignettes travaillent tous
    sur la frame réduite).
    """
    hauteur, largeur = frame.shape[:2]
    if largeur <= LARGEUR_MAX_FLUX:
        return frame
    echelle = LARGEUR_MAX_FLUX / largeur
    return cv2.resize(frame, (LARGEUR_MAX_FLUX, int(hauteur * echelle)), interpolation=cv2.INTER_AREA)


def frame_message(texte: str) -> np.ndarray:
    """Fabrique une frame de statut (source absente, modèle en chargement…).

    Les polices Hershey d'OpenCV ne gèrent que l'ASCII : pas d'accents ici.
    """
    img = np.full((480, 854, 3), (10, 14, 10), dtype=np.uint8)
    (largeur_texte, _), _ = cv2.getTextSize(texte, POLICE, 0.7, 2)
    cv2.putText(img, texte, ((854 - largeur_texte) // 2, 245), POLICE, 0.7, COULEUR_PERSONNE, 2, cv2.LINE_AA)
    return img


def extraire_boites(resultat) -> list[tuple[int, int, int, int, str, float]]:
    """Convertit un résultat ultralytics en liste (x1, y1, x2, y2, classe, conf)."""
    boites = []
    for boite in resultat.boxes:
        x1, y1, x2, y2 = (int(v) for v in boite.xyxy[0].tolist())
        nom = resultat.names[int(boite.cls[0])]
        boites.append((x1, y1, x2, y2, nom, float(boite.conf[0])))
    return boites


def dessiner_detections(frame: np.ndarray, boites: list) -> None:
    """Dessine les boîtes et leurs étiquettes (classe + confiance) sur la frame."""
    for x1, y1, x2, y2, nom, conf in boites:
        couleur = COULEUR_PERSONNE if nom == "person" else COULEUR_AUTRE
        cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
        etiquette = f"{nom} {conf:.2f}"
        (largeur, hauteur), _ = cv2.getTextSize(etiquette, POLICE, 0.5, 1)
        # Étiquette au-dessus de la boîte, ou en dessous du bord si trop haut.
        y_texte = y1 - 6 if y1 - hauteur - 10 >= 0 else y1 + hauteur + 8
        cv2.rectangle(frame, (x1, y_texte - hauteur - 4), (x1 + largeur + 6, y_texte + 4), couleur, -1)
        cv2.putText(frame, etiquette, (x1 + 3, y_texte), POLICE, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def redimensionner_vignette(crop: np.ndarray) -> np.ndarray:
    """Ramène un crop à TAILLE_VIGNETTE px de côté max (jamais agrandi)."""
    hauteur, largeur = crop.shape[:2]
    echelle = TAILLE_VIGNETTE / max(hauteur, largeur)
    if echelle >= 1.0:
        return crop
    return cv2.resize(crop, (max(1, int(largeur * echelle)), max(1, int(hauteur * echelle))), interpolation=cv2.INTER_AREA)


def capturer_vignettes(frame: np.ndarray, boites: list) -> None:
    """Extrait les crops des objets détectés et les range en tête de galerie.

    Appelée au plus une fois par seconde (INTERVALLE_VIGNETTES) sur la frame
    « propre » (avant annotation) : un même objet ne produit donc jamais plus
    d'une vignette par seconde — sans nécessiter de tracking.
    """
    hauteur, largeur = frame.shape[:2]
    for x1, y1, x2, y2, nom, conf in boites:
        # Coordonnées bornées à l'image, boîtes dégénérées ignorées.
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(largeur, x2), min(hauteur, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        vignette = redimensionner_vignette(frame[y1:y2, x1:x2])
        ok, tampon = cv2.imencode(".jpg", vignette, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        entree = {
            "id": next(etat.ids),
            "classe": nom,
            "confiance": round(conf, 2),
            "timestamp": time.strftime("%H:%M:%S"),
            "image": base64.b64encode(tampon).decode("ascii"),
        }
        with etat.verrou:
            etat.vignettes.appendleft(entree)  # deque(maxlen) évince les anciennes


def publier(frame: np.ndarray, fps: float = 0.0, nb_objets: int = 0) -> None:
    """Encode la frame en JPEG et la publie pour les clients MJPEG."""
    ok, tampon = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITE_JPEG])
    if not ok:
        return
    with etat.verrou:
        etat.frame_jpeg = tampon.tobytes()
        etat.fps = fps
        etat.nb_objets = nb_objets
        etat.frame_prete.notify_all()


# ---------------------------------------------------------------------------
# Boucle d'inférence (thread dédié)
# ---------------------------------------------------------------------------


def boucle_inference() -> None:
    """Cœur de l'application : lecture vidéo → YOLO → annotation → publication.

    Un seul thread fait tourner l'inférence, quel que soit le nombre de
    clients connectés au flux ; les endpoints ne font que lire l'état partagé.
    """
    modeles_charges: dict[str, YOLO] = {}  # cache : re-changer de modèle est instantané
    modele: YOLO | None = None
    nom_modele: str | None = None
    cap: cv2.VideoCapture | None = None
    source_courante: str | None = None  # identifiant de la source dont `cap` est issue
    echec_source_signale = False  # pour ne pas répéter le même avertissement toutes les 2 s
    delai_frame = 1 / 30
    fps_lisse = 0.0
    derniere_passe_vignettes = 0.0

    while not etat.arret.is_set():
        debut = time.perf_counter()

        # --- Chargement initial ou changement de modèle à chaud -------------
        with etat.verrou:
            demande = etat.modele_demande or etat.modele_actuel
        if demande != nom_modele:
            if modele is None:
                publier(frame_message("CHARGEMENT DU MODELE..."))
            try:
                if demande not in modeles_charges:
                    modeles_charges[demande] = charger_modele(demande)
                modele = modeles_charges[demande]
                nom_modele = demande
                fps_lisse = 0.0  # la moyenne repart de zéro avec le nouveau modèle
                with etat.verrou:
                    etat.modele_actuel = demande
                    etat.modele_demande = None
            except Exception as exc:
                logger.error("Échec du chargement de « %s » : %s", demande, exc)
                with etat.verrou:
                    etat.modele_demande = None
                if modele is None:  # aucun modèle utilisable : on réessaie plus tard
                    publier(frame_message("ERREUR : MODELE INDISPONIBLE"))
                    etat.arret.wait(2.0)
                continue

        # --- Changement de source vidéo à chaud -----------------------------
        with etat.verrou:
            source_visee = etat.source_demandee or etat.source_actuelle
        if source_visee != source_courante:
            if source_courante is not None:
                logger.info("Changement de source : « %s » → « %s »", source_courante, source_visee)
            if cap is not None:
                cap.release()  # fermeture propre de l'ancienne capture
                cap = None
            source_courante = source_visee
            echec_source_signale = False
            fps_lisse = 0.0  # la moyenne repart de zéro avec la nouvelle source
            with etat.verrou:
                etat.source_actuelle = source_visee
                etat.source_demandee = None

        # --- Ouverture (ou réouverture) de la source vidéo ------------------
        if cap is None or not cap.isOpened():
            cap, delai_frame = ouvrir_source(source_courante)
            if cap is None:
                # Source absente ou illisible (webcam débranchée, fichier
                # corrompu…) : frame de statut puis nouvel essai — l'app reste
                # stable et continue de réagir aux changements de source.
                if not echec_source_signale:
                    logger.warning("Source « %s » indisponible — nouvel essai toutes les 2 s", source_courante)
                    echec_source_signale = True
                texte = "CAMERA INDISPONIBLE" if source_courante == SOURCE_CAMERA else "SOURCE VIDEO INDISPONIBLE - NOUVEL ESSAI..."
                publier(frame_message(texte))
                etat.arret.wait(2.0)
                continue
            echec_source_signale = False

        # --- Lecture d'une frame, avec rebouclage en fin de vidéo -----------
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # la vidéo repart du début
            ok, frame = cap.read()
            if not ok:  # source non « rembobinable » (caméra coupée…) : réouverture
                cap.release()
                cap = None
                continue

        # --- Inférence + annotation + publication ---------------------------
        try:
            frame = reduire_si_besoin(frame)
            resultat = modele.predict(frame, conf=SEUIL_CONFIANCE, device=DEVICE, verbose=False)[0]
            boites = extraire_boites(resultat)

            # Vignettes : au plus une passe par seconde, sur la frame propre.
            if boites and debut - derniere_passe_vignettes >= INTERVALLE_VIGNETTES:
                capturer_vignettes(frame, boites)
                derniere_passe_vignettes = debut

            dessiner_detections(frame, boites)

            duree = time.perf_counter() - debut
            fps_instantane = 1.0 / duree if duree > 0 else 0.0
            # Moyenne mobile exponentielle : affichage stable, réactif au changement.
            fps_lisse = fps_instantane if fps_lisse == 0.0 else fps_lisse * 0.9 + fps_instantane * 0.1
            publier(frame, fps=fps_lisse, nb_objets=len(boites))
        except Exception:
            logger.exception("Erreur pendant l'inférence — reprise dans 1 s")
            etat.arret.wait(1.0)
            continue

        # --- Cadence : on ne diffuse pas plus vite que la vidéo d'origine ---
        reste = delai_frame - (time.perf_counter() - debut)
        if reste > 0:
            etat.arret.wait(reste)  # interruptible par l'arrêt du serveur

    if cap is not None:
        cap.release()
    logger.info("Boucle d'inférence arrêtée proprement.")


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Démarre le thread d'inférence avec le serveur, l'arrête avec lui."""
    thread = threading.Thread(target=boucle_inference, name="inference-yolo", daemon=True)
    thread.start()
    logger.info("Detection Viewer démarré — device : %s", DEVICE)
    yield
    etat.arret.set()
    with etat.frame_prete:
        etat.frame_prete.notify_all()  # réveille les clients MJPEG en attente
    thread.join(timeout=5)


app = FastAPI(title="Sentinel — Detection Viewer", lifespan=lifespan)

# CORS ouvert : le front est autonome et peut être hébergé sur un site externe
# sans modifier le backend — il ne consomme que les endpoints ci-dessous.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RequeteModele(BaseModel):
    """Corps attendu par POST /set_model."""

    model: str


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Sert la page web (fichier autonome, relu à chaque requête : pratique en dev)."""
    return (DOSSIER_APP / "templates" / "index.html").read_text(encoding="utf-8")


def generer_flux_mjpeg():
    """Générateur MJPEG : encadre chaque JPEG publié avec la frontière multipart."""
    while not etat.arret.is_set():
        with etat.frame_prete:
            etat.frame_prete.wait(timeout=1.0)  # réveillé à chaque nouvelle frame
            frame = etat.frame_jpeg
        if frame is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n" + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii") + frame + b"\r\n"
        )


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    """Flux vidéo annoté, lisible par une simple balise <img>."""
    return StreamingResponse(generer_flux_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detections")
async def detections() -> dict:
    """Vignettes des derniers objets détectés, les plus récentes en premier."""
    with etat.verrou:
        return {"detections": list(etat.vignettes)}


@app.get("/stats")
async def stats() -> dict:
    """Statistiques temps réel affichées dans la barre du haut."""
    with etat.verrou:
        return {
            "fps": round(etat.fps, 1),
            "modele": etat.modele_actuel,
            "modele_en_chargement": etat.modele_demande,
            "device": DEVICE,
            "objets": etat.nb_objets,
            "source": etat.source_actuelle,
            "source_en_chargement": etat.source_demandee,
            "modeles": list(MODELES),
        }


@app.post("/set_model")
async def set_model(requete: RequeteModele) -> dict:
    """Demande un changement de modèle ; le thread d'inférence le recharge à chaud."""
    nom = requete.model.strip().lower().removesuffix(".pt")
    if nom not in MODELES:
        raise HTTPException(status_code=400, detail=f"Modèle inconnu : {requete.model!r}. Choix : {', '.join(MODELES)}")
    with etat.verrou:
        deja_actif = nom == etat.modele_actuel and etat.modele_demande is None
        if not deja_actif:
            etat.modele_demande = nom
    return {"ok": True, "modele": nom, "statut": "déjà actif" if deja_actif else "chargement en cours"}


@app.get("/sources")
async def sources() -> dict:
    """Sources vidéo disponibles — le dossier videos_test/ est re-scanné à chaque appel."""
    with etat.verrou:
        actuelle, demandee = etat.source_actuelle, etat.source_demandee
    return {"sources": lister_sources(), "source_actuelle": actuelle, "source_en_chargement": demandee}


class RequeteSource(BaseModel):
    """Corps attendu par POST /set_source."""

    source: str


@app.post("/set_source")
async def set_source(requete: RequeteSource) -> dict:
    """Demande un changement de source ; le thread d'inférence rouvre la capture à chaud."""
    ident = requete.source.strip()
    # Validation stricte contre la liste connue : jamais de chemin arbitraire.
    if ident not in {s["id"] for s in lister_sources()}:
        raise HTTPException(status_code=400, detail=f"Source inconnue : {requete.source!r}")
    with etat.verrou:
        deja_active = ident == etat.source_actuelle and etat.source_demandee is None
        if not deja_active:
            etat.source_demandee = ident
    return {"ok": True, "source": ident, "statut": "déjà active" if deja_active else "changement en cours"}


if __name__ == "__main__":
    # Lancement direct possible : python tools/detection_viewer/app.py
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8100)
