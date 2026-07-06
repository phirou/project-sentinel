# Detection Viewer

Outil web local de visualisation temps réel des détections YOLO du projet
Sentinel : le backend fait tourner un modèle ultralytics sur une vidéo lue en
boucle, diffuse le flux annoté en MJPEG et alimente une galerie de vignettes
des objets détectés.

## Fonctionnalités

- Flux vidéo annoté en direct (MJPEG, lisible par une simple balise `<img>`) ;
- Boîtes de détection colorées : vert phosphore pour `person`, orange pour les
  autres classes, avec étiquette « classe + ID de tracking + confiance » ;
- **Tracking multi-objets (ByteTrack, inclus dans ultralytics)** : chaque objet
  reçoit un ID stable entre les frames (ex. `person #4`), avec une courte trace
  de trajectoire. Un **compteur d'objets uniques par classe** (basé sur les IDs
  distincts vus, pas le cumul de détections) s'affiche dans la barre du haut, et
  un bouton **Reset** remet les compteurs à zéro. Le tracking est **activable /
  désactivable** (toggle « détection simple ↔ tracking ») pour comparer ;
- Changement de modèle **à chaud** (yolov8n / yolov8s / yolo11n), sans
  redémarrer le serveur ;
- Changement de **source vidéo à chaud** : vidéos du dossier `videos_test/`
  (re-scanné à chaque appel de `/sources` — y déposer un fichier suffit) ou
  webcam locale, entrée préparée pour la Pi Camera / un flux RTSP ;
- Robustesse : webcam absente ou fichier illisible → frame de statut et
  nouvel essai toutes les 2 s, sans crash ;
- Galerie des dernières détections : vignettes extraites au plus une fois par
  seconde, 20 conservées en mémoire, les plus récentes en premier ;
- FPS d'inférence en temps réel, device auto-détecté (MPS / CUDA / CPU).

## Lancement

Depuis la racine du repo :

```bash
source .venv/bin/activate
uvicorn tools.detection_viewer.app:app --port 8100
```

puis ouvrir <http://localhost:8100>. Le port 8100 évite tout conflit avec
Sentinel (port 8000). Arrêt : `Ctrl+C` (deux fois si un flux est encore ouvert
dans un onglet).

## API

| Méthode | Endpoint      | Rôle                                                    |
| ------- | ------------- | ------------------------------------------------------- |
| GET     | `/`           | Page web                                                 |
| GET     | `/video_feed` | Flux MJPEG annoté (`multipart/x-mixed-replace`)          |
| GET     | `/detections` | Vignettes récentes (JSON : image base64, classe, confiance, heure) |
| GET     | `/stats`      | FPS, modèle et source courants, device, nb d'objets, mode tracking, objets uniques par classe |
| GET     | `/sources`    | Sources vidéo disponibles (`videos_test/` + caméra)      |
| POST    | `/set_model`  | Changement de modèle, corps `{"model": "yolov8s"}`       |
| POST    | `/set_source` | Changement de source, corps `{"source": "camera"}`       |
| POST    | `/set_tracking` | Active/désactive le tracking, corps `{"enabled": true}` |
| POST    | `/reset_counts` | Remet à zéro les compteurs d'objets uniques            |

La documentation interactive FastAPI est disponible sur `/docs`.

## Architecture

- **Un seul thread d'inférence** (lecture vidéo → YOLO → annotation OpenCV →
  encodage JPEG), quel que soit le nombre de clients connectés : les endpoints
  ne font que lire un état partagé protégé par verrou.
- Les clients MJPEG sont réveillés par une `Condition` à chaque nouvelle
  frame : pas d'attente active.
- **Tracking** : ByteTrack tourne dans le même thread via
  `model.track(persist=True)` ; le tracker et les compteurs sont réinitialisés
  proprement à chaque changement de source, de modèle ou de mode (une passe
  `persist=False` réinitialise ByteTrack). Les compteurs d'objets uniques sont
  des `set` d'IDs par classe, protégés par le verrou d'état.
- **Front 100 % autonome** (HTML/CSS/JS vanilla, aucun framework) : la page ne
  dépend du backend qu'à travers ses endpoints (voir table ci-dessus). Pour
  l'héberger sur un site externe, renseigner la constante `API_BASE` en tête du
  script de `templates/index.html` — le backend autorise déjà le CORS.

## Configuration

Les réglages sont des constantes en tête de `app.py` : `VIDEO_SOURCE` (source
par défaut au démarrage), `DOSSIER_VIDEOS`, `CIBLE_CAMERA` (index webcam,
remplaçable par une URL RTSP pour la Pi Camera), `MODELES`, `MODELE_PAR_DEFAUT`
(`yolo11n`, retenu à l'issue du benchmark), `SEUIL_CONFIANCE`, et pour le
tracking : `MODE_TRACKING_PAR_DEFAUT`, `TRACKER_YAML`, `LONGUEUR_TRACE`,
`SEUIL_SAUT_TRACE` (seuil anti-lignes parasites des traces), `INTERVALLE_VIGNETTES`,
`MAX_VIGNETTES`, couleurs, etc.
