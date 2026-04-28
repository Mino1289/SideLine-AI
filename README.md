# SideLine-AI

Projet d'apprentissage profond pour l'analyse de vidéos de hockey sur glace. Le système permet la détection, le suivi, l'identification des joueurs par reconnaissance optique de caractères (OCR) et la classification des équipes, ainsi que la projection des positions sur une vue 2D de la patinoire.

## Fonctionnalités principales

*   **Détection et segmentation** : Utilisation de modèles YOLO (PyTorch et OpenVINO) pour identifier les joueurs, les arbitres et la rondelle.
*   **Suivi (Tracking)** : Suivi des entités détectées à travers les trames vidéo à l'aide de ByteTrack (via la bibliothèque supervision).
*   **Classification d'équipes** : Algorithme de séparation des équipes basé sur l'extraction des couleurs des uniformes.
*   **Identification (OCR)** : Reconnaissance des numéros de joueurs via EasyOCR couplé à un suivi d'historique pour fiabiliser les prédictions au fil du temps.
*   **Cartographie 2D** : Utilisation de modèles de détection de points clés (keypoints) pour calculer une matrice d'homographie et projeter les coordonnées vidéo sur un schéma 2D de la patinoire.

## Structure du projet

*   `main.py` : Point d'entrée pour lancer le traitement vidéo complet.
*   `config.py` : Configuration globale (chemins, coordonnées de la patinoire, composition et couleurs des équipes).
*   `core/`
    *   `pipeline.py` : Orchestrateur central qui gère le flux de traitement trame par trame.
*   `vision/`
    *   `detector.py` : Gestion des inférences YOLO pour la segmentation et les points clés.
    *   `identifier.py` : Reconnaissance optique et lissage temporel des numéros de joueurs.
    *   `mapper.py` : Transformations spatiales (homographie, géométrie).
    *   `team_classifier.py` : Classification des couleurs.
*   `utils/`
    *   `annotator.py` : Fonctions pour le tracé graphique sur la vidéo et la patinoire virtuelle.
*   `models/` : Emplacement des poids des réseaux de neurones (`.pt` et formats OpenVINO).

## Modèles

Téléchargez le modèle HockeyAI depuis [SimulaMet-HOST/HockeyAI](https://huggingface.co/SimulaMet-HOST/HockeyAI) et placez-le dans le dossier `models/`.
Les autres modèles YOLO et EasyOCR seront normalement téléchargés automatiquement lors de l'exécution.

## Prérequis et installation

Assurez-vous d'avoir Python d'installé (de préférence dans un environnement virtuel).

1.  Installer les dépendances requises :

    ```bash
    pip install -r requirements.txt
    ```

Les bibliothèques principales requises sont `ultralytics`, `supervision`, `easyocr`, `opencv-python`, et `scikit-learn`.

## Utilisation

1.  Configuration : Modifiez `config.py` pour indiquer les chemins de vos vidéos d'entrée (`SOURCE_VIDEO_PATH`) et le répertoire de sortie (`OUT_VIDEO_DIR`). Vous pouvez également y définir les paramètres spécifiques au match (joueurs, équipes).
2.  Lancement du traitement :

    ```bash
    python main.py
    ```

La vidéo traitée, incluant les détections et la cartographie, sera générée dans le dossier de sortie défini.