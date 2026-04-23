import os
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
import numpy as np

class HockeyDetector:
    """Gère la détection (Segmentation et Keypoints) via YOLO."""
    
    def __init__(self, seg_model_path: Path, seg_fallback_path: Path, ov_seg_path: Path, kp_model_path: Path, ov_kp_path: Path):
        
        # Logique d'initialisation YOLO_SEG
        if os.path.exists(ov_seg_path):
            self.seg_model = YOLO(ov_seg_path, task="segment")
        else:
            final_seg_path = seg_model_path if seg_model_path.exists() else seg_fallback_path
            self.seg_model = YOLO(final_seg_path, task="segment")
            
        # Logique d'initialisation YOLO_KP
        if os.path.exists(ov_kp_path):
            self.kp_model = YOLO(ov_kp_path, task="detect")
        else:
            self.kp_model = YOLO(kp_model_path, task="detect")
        
        # Résolution du class ID du joueur
        self.player_class_id = next(
            (k for k, v in self.seg_model.names.items() if v.lower() in ["person", "player"]), 0
        )
        self.puck_class_id = next(
            (k for k, v in self.kp_model.names.items() if v.lower() == "puck"), -1
        )

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        results = self.seg_model.predict(frame, verbose=False, imgsz=640)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections[detections.class_id == self.player_class_id]

    def detect_keypoints_and_puck(self, frame: np.ndarray):
        """Retourne les points clés du terrain et la rondelle."""
        results = self.kp_model.predict(frame, verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        if self.puck_class_id != -1:
            puck = detections[detections.class_id == self.puck_class_id]
        else:
            puck = sv.Detections.empty()
            
        return detections, puck
