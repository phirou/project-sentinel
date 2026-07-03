# Benchmark des modèles YOLO — Project Sentinel

*Généré le 2026-07-03T09:23:24 — device : **MPS**, FRAME_STRIDE : 3, conf : 0.25.*

> ⚠️ **Cible réelle : Raspberry Pi 5 en CPU.** Ce benchmark tourne sur `MPS`. Les **FPS absolus ne sont pas transférables** au Pi ; en revanche le **classement relatif** des modèles et **toutes les métriques de détection** (sensibilité, confiance, nombre de détections) le sont, car elles ne dépendent pas du matériel. Pour des FPS au caractère CPU : `--device cpu`.

## Résumé par modèle et résolution

Agrégé sur toutes les vidéos. `%f pers` = part des frames contenant au moins une personne (proxy de sensibilité) ; `Pers/f` = nombre moyen de personnes par frame.

| Modèle | imgsz | FPS | Dét. tot. | Pers/f | %f pers | Conf. moy. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n.pt | 640 | 32.4 | 5820 | 4.09 | 91.0 | 0.525 |
| yolov8n.pt | 640 | 26.5 | 6467 | 5.06 | 95.7 | 0.529 |
| yolov8s.pt | 640 | 23.0 | 6891 | 4.33 | 98.3 | 0.535 |
| yolo11n.pt | 1280 | 15.2 | 12368 | 10.63 | 98.2 | 0.569 |
| yolov8n.pt | 1280 | 15.1 | 15267 | 13.65 | 98.1 | 0.572 |
| yolov8s.pt | 1280 | 13.9 | 14552 | 11.60 | 98.4 | 0.581 |

## Détail par vidéo

| Modèle | imgsz | Vidéo | FPS | Dét. | Pers/f | %f pers | Conf. |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| yolo11n.pt | 640 | crowd_street_dense | 23.5 | 2941 | 10.51 | 100.0 | 0.534 |
| yolo11n.pt | 640 | forest_2people_2dogs_tracking | 38.9 | 201 | 1.47 | 100.0 | 0.646 |
| yolo11n.pt | 640 | forest_2people_hiking | 55.1 | 115 | 1.39 | 76.6 | 0.603 |
| yolo11n.pt | 640 | forest_autumn_2people_path | 47.4 | 586 | 2.02 | 100.0 | 0.694 |
| yolo11n.pt | 640 | street_cars_pedestrians_far | 23.8 | 1977 | 0.71 | 59.0 | 0.444 |
| yolo11n.pt | 1280 | crowd_street_dense | 9.3 | 8872 | 32.67 | 100.0 | 0.557 |
| yolo11n.pt | 1280 | forest_2people_2dogs_tracking | 22.4 | 243 | 1.60 | 100.0 | 0.656 |
| yolo11n.pt | 1280 | forest_2people_hiking | 24.1 | 148 | 1.42 | 80.5 | 0.703 |
| yolo11n.pt | 1280 | forest_autumn_2people_path | 26.7 | 591 | 2.00 | 100.0 | 0.786 |
| yolo11n.pt | 1280 | street_cars_pedestrians_far | 12.3 | 2514 | 2.14 | 100.0 | 0.546 |
| yolov8n.pt | 640 | crowd_street_dense | 17.4 | 3613 | 13.47 | 100.0 | 0.511 |
| yolov8n.pt | 640 | forest_2people_2dogs_tracking | 34.6 | 207 | 1.52 | 100.0 | 0.708 |
| yolov8n.pt | 640 | forest_2people_hiking | 42.9 | 117 | 1.51 | 80.5 | 0.669 |
| yolov8n.pt | 640 | forest_autumn_2people_path | 41.6 | 579 | 2.00 | 100.0 | 0.738 |
| yolov8n.pt | 640 | street_cars_pedestrians_far | 21.6 | 1951 | 1.38 | 84.9 | 0.472 |
| yolov8n.pt | 1280 | crowd_street_dense | 8.1 | 11500 | 42.77 | 100.0 | 0.569 |
| yolov8n.pt | 1280 | forest_2people_2dogs_tracking | 22.4 | 302 | 1.89 | 100.0 | 0.657 |
| yolov8n.pt | 1280 | forest_2people_hiking | 30.9 | 140 | 1.40 | 79.2 | 0.708 |
| yolov8n.pt | 1280 | forest_autumn_2people_path | 32.2 | 690 | 2.15 | 100.0 | 0.742 |
| yolov8n.pt | 1280 | street_cars_pedestrians_far | 13.3 | 2635 | 2.54 | 100.0 | 0.524 |
| yolov8s.pt | 640 | crowd_street_dense | 16.1 | 3601 | 10.74 | 100.0 | 0.465 |
| yolov8s.pt | 640 | forest_2people_2dogs_tracking | 28.0 | 244 | 1.53 | 100.0 | 0.735 |
| yolov8s.pt | 640 | forest_2people_hiking | 36.9 | 111 | 1.42 | 81.8 | 0.747 |
| yolov8s.pt | 640 | forest_autumn_2people_path | 35.4 | 578 | 2.00 | 100.0 | 0.772 |
| yolov8s.pt | 640 | street_cars_pedestrians_far | 17.4 | 2357 | 1.73 | 100.0 | 0.555 |
| yolov8s.pt | 1280 | crowd_street_dense | 9.0 | 10652 | 35.03 | 100.0 | 0.559 |
| yolov8s.pt | 1280 | forest_2people_2dogs_tracking | 16.6 | 324 | 1.90 | 100.0 | 0.721 |
| yolov8s.pt | 1280 | forest_2people_hiking | 20.4 | 152 | 1.44 | 83.1 | 0.778 |
| yolov8s.pt | 1280 | forest_autumn_2people_path | 20.5 | 670 | 2.00 | 100.0 | 0.745 |
| yolov8s.pt | 1280 | street_cars_pedestrians_far | 13.7 | 2754 | 3.71 | 100.0 | 0.596 |

## Répartition des détections par classe

Agrégée sur toutes les vidéos, confiance moyenne entre parenthèses.

| Modèle | imgsz | Classes détectées (nb, conf. moy.) |
| :--- | ---: | :--- |
| yolo11n.pt | 640 | person : 3418 (0.59) · car : 1517 (0.46) · truck : 292 (0.41) · handbag : 272 (0.34) · clock : 91 (0.34) · traffic light : 78 (0.43) · dog : 59 (0.46) · potted plant : 53 (0.37) · bus : 12 (0.37) · backpack : 10 (0.30) · boat : 6 (0.37) · fire hydrant : 5 (0.47) · bird : 3 (0.40) · umbrella : 2 (0.36) · giraffe : 1 (0.34) · sheep : 1 (0.26) |
| yolov8n.pt | 640 | person : 4221 (0.56) · car : 1325 (0.47) · truck : 397 (0.49) · handbag : 213 (0.34) · traffic light : 165 (0.47) · dog : 58 (0.62) · bus : 45 (0.34) · clock : 19 (0.34) · potted plant : 9 (0.30) · backpack : 6 (0.32) · giraffe : 3 (0.30) · cow : 2 (0.45) · bench : 1 (0.26) · sheep : 1 (0.29) · bear : 1 (0.84) · motorcycle : 1 (0.46) |
| yolov8s.pt | 640 | person : 3615 (0.55) · car : 1558 (0.59) · handbag : 773 (0.40) · truck : 376 (0.56) · traffic light : 175 (0.45) · potted plant : 137 (0.44) · dog : 96 (0.72) · horse : 63 (0.33) · clock : 45 (0.32) · bus : 25 (0.43) · backpack : 8 (0.32) · elephant : 7 (0.30) · motorcycle : 6 (0.27) · bottle : 2 (0.36) · frisbee : 1 (0.29) · bird : 1 (0.37) · giraffe : 1 (0.30) · baseball glove : 1 (0.39) · parking meter : 1 (0.28) |
| yolo11n.pt | 1280 | person : 8875 (0.60) · car : 1591 (0.59) · handbag : 871 (0.38) · truck : 457 (0.49) · traffic light : 203 (0.42) · bus : 120 (0.32) · potted plant : 60 (0.34) · backpack : 56 (0.40) · dog : 50 (0.48) · elephant : 32 (0.50) · bird : 19 (0.37) · horse : 7 (0.33) · bicycle : 7 (0.32) · cow : 6 (0.38) · bear : 5 (0.44) · sports ball : 4 (0.37) · skateboard : 2 (0.36) · fire hydrant : 2 (0.28) · tennis racket : 1 (0.39) |
| yolov8n.pt | 1280 | person : 11394 (0.61) · car : 1597 (0.56) · handbag : 865 (0.41) · truck : 519 (0.47) · traffic light : 341 (0.34) · bus : 116 (0.34) · dog : 79 (0.65) · backpack : 64 (0.36) · motorcycle : 60 (0.35) · refrigerator : 45 (0.38) · parking meter : 43 (0.37) · horse : 19 (0.32) · tennis racket : 16 (0.45) · cow : 15 (0.35) · bird : 14 (0.44) · suitcase : 13 (0.31) · clock : 11 (0.33) · elephant : 10 (0.32) · potted plant : 9 (0.28) · boat : 9 (0.46) · umbrella : 5 (0.32) · dining table : 5 (0.28) · giraffe : 5 (0.35) · bear : 3 (0.37) · oven : 3 (0.28) · sports ball : 2 (0.53) · fire hydrant : 2 (0.48) · skateboard : 1 (0.34) · zebra : 1 (0.27) · cat : 1 (0.33) |
| yolov8s.pt | 1280 | person : 9684 (0.59) · handbag : 1858 (0.49) · car : 1627 (0.66) · truck : 502 (0.57) · traffic light : 352 (0.62) · motorcycle : 128 (0.46) · dog : 102 (0.75) · backpack : 98 (0.50) · potted plant : 62 (0.35) · oven : 41 (0.34) · bus : 35 (0.36) · bicycle : 18 (0.34) · clock : 9 (0.29) · bird : 9 (0.47) · stop sign : 9 (0.34) · bear : 5 (0.46) · frisbee : 3 (0.34) · horse : 3 (0.38) · cell phone : 2 (0.43) · umbrella : 1 (0.25) · giraffe : 1 (0.50) · zebra : 1 (0.26) · sports ball : 1 (0.40) · parking meter : 1 (0.26) |

## Méthodologie

- **Modèles** : yolov8n.pt, yolov8s.pt, yolo11n.pt.
- **Résolutions** (`imgsz`) : 640, 1280.
- **Échantillonnage** : 1 frame sur 3 (le même pour tous → comparaison équitable, exactement les mêmes frames comparées).
- **Seuil de confiance** : 0.25 (constant).
- **FPS** : temps de la seule inférence `predict()` (décodage vidéo exclu), avec synchronisation du device pour une mesure fiable.
- Une frame annotée par modèle × vidéo est disponible dans `samples/`.
