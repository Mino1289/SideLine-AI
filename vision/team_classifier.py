import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Optional

def extract_color_histogram(image: np.ndarray, mask: Optional[np.ndarray] = None, bins: tuple = (8, 8, 8)) -> np.ndarray:
    """Extrait un histogramme de couleur (HSV) standardisé pour une image (et un masque)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calcul de l'histogramme HSV
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def cluster_teams(crops: List[np.ndarray], masks: Optional[List[np.ndarray]] = None, n_clusters: int = 2) -> np.ndarray:
    """Clustérise une liste de crops de joueurs en 'n' équipes (généralement 2) basées sur la couleur."""
    features = []
    for i, crop in enumerate(crops):
        mask = masks[i] if masks is not None else None
        
        # Convertir le masque (boolean/floats) en uint8 pour OpenCV si nécessaire
        if mask is not None and mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
            
        hist = extract_color_histogram(crop, mask)
        features.append(hist)
    
    if not features:
        return np.array([])
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels
