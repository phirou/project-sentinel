# Sentinel — Journal de projet

> Mémoire vivante du projet. À relire (ou recoller à un assistant) en début de
> session pour reprendre le contexte sans rien oublier. Mettre à jour à chaque
> avancée importante.

Dernière mise à jour : 2026-05-22

---

## 0. Pitch & objectif

Système de surveillance autonome inspiré des Sentry Towers d'Anduril.
**Détection uniquement, aucun armement.** Workflow : radar mmWave détecte du
mouvement 360° → caméra pointe vers la cible → IA embarquée classifie
(humain / animal / véhicule). Dashboard tactique temps réel.

Objectif personnel : projet portfolio pour viser l'ingénierie défense/aéro
(Anduril, Thales, MBDA, Dassault, SpaceX).

---

## 1. État du projet (phases)

- **Phase 1 — Détecteur radar fixe + dashboard temps réel** : ✅ FONCTIONNEL
  Radar physique validé, déployé sur Pi embarqué autonome.
- **Phase 2 — Caméra + IA + auto-pointage** : ⏳ EN COURS
  Caméra branchée + capture déclenchée par radar (slew-to-cue) ✅.
  IA de classification : en cours de développement.
- **Phase 3 — Longue portée + fusion + tracking multi-cibles** : 🔜 planifié.

---

## 2. Hardware

### Composants
- Raspberry Pi 5 (8 GB) + Active Cooler + alim 27W officielle.
- Radar mmWave **Hi-Link HLK-LD2450** (24 GHz, jusqu'à 3 cibles, distance + angle).
- Module **HKJL** (breakout/convertisseur du LD2450) + câble JST 4 fils.
- Convertisseur USB-UART **HLK-CH340E**.
- **Pi Camera Module 3** (capteur Sony IMX708) via nappe CSI (port J3 / CAM/DISP 0).

### Câblage radar (convention couleur INVERSÉE Hi-Link — vérifiée physiquement)
| Fil JST | Signal radar | Pin CH340E |
|---|---|---|
| Noir | 5V | 5V (broche latérale) |
| Jaune | GND | GND |
| Blanc | TX radar | RXT (croisé) |
| Rouge | RX radar | TXD (croisé) |

⚠️ Noir = 5V et Jaune = GND : inverse de la convention habituelle, mais c'est
celle de Hi-Link. TX/RX croisés (règle UART).

### Pièges hardware rencontrés / à retenir
- Ne JAMAIS débrancher le Pi sans `sudo shutdown -h now` (corruption carte SD).
- Pi 5 : une seule LED « STAT ». Vert = fonctionne, **rouge fixe = halté** (OK pour débrancher).
- Nappe CSI Pi 5 = connecteur 15 broches (différent des Pi précédents, d'où le câble 22→15).
- Brancher la nappe caméra Pi ÉTEINT uniquement.
- Plus tard (servos pan/tilt) : alim séparée pour les servos, jamais sur le 5V du Pi.

### Ports
- Sur le Mac : radar = `/dev/cu.usbserial-11240`.
- Sur le Pi : radar = `/dev/ttyUSB0` (nécessite `usermod -aG dialout leo` une fois).

---

## 3. Architecture logicielle

Repo : github.com/phirou/project-sentinel (public, MIT).
Local Mac : `~/Developer/project-sentinel` — Local Pi : `~/project-sentinel`.

```
sentinel/
  core/        models.py, event_bus.py, persistence.py, queries.py
  drivers/     ld2450_parser.py, ld2450_simulator.py, ld2450_driver.py, camera.py
  consumers/   camera_trigger.py
  api/         app.py (FastAPI + WebSocket)
  main.py      orchestration asyncio
web/           dashboard tactique (HTML/CSS/JS)
ml/            dataset.py, transforms.py, model.py, train.py, config.yaml
docs/          architecture.md, JOURNAL.md
```

### Principes d'archi
- **EventBus pub/sub** : les producteurs (drivers) publient, les consommateurs
  (stats, console, persistance SQLite, caméra) s'abonnent. Ajouter un consumer
  ne touche pas au pipeline.
- **Inversion de dépendance** radar : simulateur et vrai driver exposent la même
  interface `stream()`. Le code ne sait pas si c'est simulé ou réel.
- **Asyncio** : pipeline radar + serveur web + stats sur une seule boucle.
  Capture caméra déportée en thread (`asyncio.to_thread`) pour ne pas bloquer.
- Persistance **SQLite** async (aiosqlite), config **YAML** (default + local).

### Bugs marquants résolus
- **Convention d'axe Y radar** : le LD2450 renvoie Y < 0 DEVANT le radar
  (inverse de la convention informatique). Sans correction, toutes les cibles
  apparaissaient à ~180°. Fix : `atan2(x, -y)` dans `models.py`. Validé par
  mesure physique (face=0°, gauche/droite cohérents).
- **Shutdown asyncio** : uvicorn interceptait SIGINT. Fix :
  `server.install_signal_handlers = lambda: None` + annulation explicite.

### Slew-to-cue (Phase 2)
`CameraTrigger` (consumer) : sur détection radar → capture caméra non-bloquante,
avec throttle (cooldown 5s) et nom de fichier encodant la position
(`T0_d2.3m_a-15deg`) pour lier détection ↔ image (futur dataset ML).

---

## 4. Décisions ML

- **Compute** : entraînement sur **Mac M3** (backend MPS, validé `MPS dispo: True`).
  Inférence finale visée sur **NPU Hailo-8** du Pi. (train sur machine puissante,
  inference sur l'embarqué = archi standard.)
- **Framework** : **PyTorch** (standard industrie défense, support Hailo).
  NB : PyTorch retiré du venv Pi (lignes ML commentées) — le Pi ne fait pas l'entraînement.
- **Approche pédagogique** : d'abord un **CNN écrit de zéro** (comprendre conv,
  pool, fully-connected), PUIS **transfer learning** (MobileNetV3) pour comparer.
- **Tâche V1** : classification **binaire humain / non-humain**. Conçue pour
  passer en multi-classes (animal, véhicule) sans refonte (`num_classes` param +
  dossiers par classe).
- **Taille d'entrée** : `IMAGE_SIZE = 128` (compromis vitesse/précision). Modifiable
  en une constante ; le CNN s'adapte via `AdaptiveAvgPool2d`.
- **Dataset** : bootstrap datasets publics (Penn-Fudan pour humains) + photos
  Sentinel perso plus tard. Léo veut construire un dataset propre **avec son père**
  (qui s'y connaît en data). Dossiers : `ml/data/{train,val}/{human,non_human}/`
  (gitignorés).

### Concepts ML déjà couverts
- Léo a suivi la formation deep learning de MachineLearnia (Guillaume Saint-Cirgue,
  backprop à la main en NumPy). A déjà fait un classifieur Pokémon (probablement Keras).
- Comprend forward/backward propagation. En PyTorch : forward = méthode `forward()`,
  backward = `loss.backward()` (autograd automatique). Pas de gradients à la main.

### État des fichiers ML (au 2026-05-22)
- `dataset.py` ✅ écrit (ImageFolder, `build_dataset`). Testé : trouve les classes,
  0 image pour l'instant (dossiers vides — normal).
- `transforms.py` ✅ écrit (train avec augmentation, val sans).
- `model.py` ✅ écrit (`SentinelCNN`, 3 blocs conv + classifieur, 155k params).
  Testé avec batch factice → sortie (4, 2) correcte.
- `train.py` ⏳ à écrire (boucle d'entraînement : forward, loss, backward, step).
- `config.yaml` ⏳ vide (hyperparamètres à externaliser).

---

## 5. Méthode de travail (préférences de Léo)

- **Pédagogie** : donner le code **par blocs** avec explications, Léo **tape
  lui-même** (pas de copier-coller aveugle, pas Claude Code pour le ML — c'est
  la compétence à acquérir). Valider chaque étape avant d'avancer.
- **Ne pas forcer les pauses** : si Léo dit qu'il continue, on continue.
- **Validation** : toujours faire exécuter/tester avant de passer à la suite
  (leçon du « clone fantôme » et des étapes sautées).
- **Quand un terminal est envoyé et que tout est bon** : réponse courte
  (« Parfait, aucun problème »), pas de pavé, pas de « next » à sa place.
- **Toujours préciser** 🍓 Pi vs 💻 Mac pour chaque commande.
- **Repère prompts** : Pi = `leo@sentinel`, Mac = `leos-Air-001`.

---

## 6. Déploiement & exploitation

- Lancer Sentinel (Pi) : `ssh leo@sentinel.local` → `cd ~/project-sentinel`
  → `source .venv/bin/activate` → `python -m sentinel.main`.
- Dashboard : `http://sentinel.local:8000` (ou IP, ex. `http://192.168.1.49:8000`).
- venv Pi recréé avec `--system-site-packages` (pour accéder à `picamera2` système).
- WiFi de Lille ajouté au Pi (utilisable en déplacement le week-end).
- systemd auto-start : **volontairement PAS activé** (Léo veut garder le contrôle
  du lancement manuel pendant le dev). À faire en fin de projet.
- Git push depuis le Pi : nécessite un Personal Access Token (PAT), pas le mot de
  passe. `credential.helper store` activé pour mémoriser.

---

## 7. Next steps

1. Écrire `train.py` (boucle d'entraînement) + `config.yaml` (hyperparamètres).
2. Construire le dataset (datasets publics + aide du père de Léo).
3. Premier entraînement réel + lecture des courbes (loss/accuracy train vs val).
4. Transfer learning (MobileNetV3) pour comparer.
5. Compilation du modèle pour le NPU Hailo (inférence sur Pi).
6. Brancher la classification dans le pipeline (consumer IA sur les captures).
7. Plus tard : +1 LD2450 (test multi-radar 360°), servos pan/tilt, dashboard mobile,
   replay UI, documentation (docs/ld2450-protocol.md).
