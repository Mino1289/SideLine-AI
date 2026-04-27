import logging
from datetime import datetime
import numpy as np
import cv2
import supervision as sv
from tqdm.auto import tqdm

import config
from vision.detector import HockeyDetector
from vision.identifier import OCRIdentifier, OCRHistoryTracker
from vision.mapper import ViewTransformer, map_generic_to_specific_points
from vision.team_classifier import cluster_teams
from utils.annotator import draw_hockey_rink_2d, draw_points_on_rink

logger = logging.getLogger(__name__)

class HockeyVideoPipeline:
    """Orchestrateur principal traitant la trame globale de la vidéo."""

    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path
        
        logger.info("Initialisation des modèles et trackers...")
        self.detector = HockeyDetector(
            seg_model_path=config.YOLO_SEG_MODEL_PATH,
            seg_fallback_path=config.YOLO_SEG_MODEL_PATH_FALLBACK,
            ov_seg_path=config.OV_SEG_MODEL_PATH,
            kp_model_path=config.YOLO_KP_MODEL_PATH,
            ov_kp_path=config.OV_KEYPOINT_MODEL_PATH
        )
        self.ocr = OCRIdentifier()
        self.ocr_tracker = OCRHistoryTracker(n_consecutive=3)
        
        self.tracker_players = sv.ByteTrack()
        self.tracker_puck = sv.ByteTrack()

    def _get_actual_frame_count(self, video_path: str) -> int:
        """Calcule le vrai nombre de frames pour remplacer les métadonnées corrompues."""
        logger.info("Analyse de la vidéo pour déterminer le nombre réel de frames...")
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.grab():
            count += 1
        cap.release()
        return count

    def run(self):
        logger.info(f"Démarrage du traitement de {self.source_path}...")
        video_info = sv.VideoInfo.from_video_path(str(self.source_path))
        
        # Obtenir le vrai total indépendamment de metadata trompeuses
        actual_total_frames = self._get_actual_frame_count(str(self.source_path))
        video_info.total_frames = actual_total_frames # Mise à jour du metadata pour la barre
        
        frame_generator = sv.get_video_frames_generator(str(self.source_path))
        
        base_rink_2d = draw_hockey_rink_2d(scale=3)
        
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.6, color=sv.Color.BLACK)
        mask_annotator = sv.MaskAnnotator()
        
        transformer = None
        last_puck = sv.Detections.empty()

        with sv.VideoSink(str(self.target_path), video_info) as sink:
            for i, frame in enumerate(tqdm(frame_generator, total=actual_total_frames, desc="Processing Video")):
                # 1. Détection et Suivi
                det_players = self.detector.detect_players(frame)
                det_players = self.tracker_players.update_with_detections(det_players)
                
                # 1.5 Classification d'équipes
                if len(det_players) > 0:
                    crops = []
                    masks = []
                    for i, xyxy in enumerate(det_players.xyxy):
                        x1, y1, x2, y2 = map(int, xyxy)
                        crops.append(frame[max(0, y1):y2, max(0, x1):x2])
                        if det_players.mask is not None:
                            masks.append(det_players.mask[i, max(0, y1):y2, max(0, x1):x2])
                    
                    team_labels = cluster_teams(crops, masks if len(masks) == len(crops) else None)
                    det_players.class_id = team_labels.astype(int)
                
                # 2. OCR (
                self._update_ocr(det_players, frame)
                
                labels = self._generate_labels(det_players)

                # 3. Points clés et Puck 
                kp_all, det_puck = self.detector.detect_keypoints_and_puck(frame)
                try:
                    last_puck = self.tracker_puck.update_with_detections(det_puck)
                except Exception: pass
                
                new_transformer = self._update_homography(kp_all, video_info.width)
                if new_transformer is not None:
                    transformer = new_transformer
                
                # 4. Rendu Visuel
                annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=det_players)
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=det_players)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=det_players, labels=labels)
                
                # Overlay 2D
                self._draw_minimap_overlay(annotated_frame, base_rink_2d, transformer, det_players, last_puck)
                
                sink.write_frame(annotated_frame)
                
        logger.info(f"Traitement terminé. Vidéo sauvegardée: {self.target_path}")

    def _update_ocr(self, detections: sv.Detections, frame: np.ndarray):
        values = []
        for tid, xyxy in zip(detections.tracker_id, detections.xyxy):
            if tid in self.ocr_tracker.validated:
                values.append(self.ocr_tracker.validated[tid])
            else:
                x1, y1, x2, y2 = map(int, xyxy)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                values.append(self.ocr.read_number(crop))
        self.ocr_tracker.update(detections.tracker_id, values)

    def _generate_labels(self, detections) -> list:
        valid_ocrs = self.ocr_tracker.get_validated(detections.tracker_id)
        return [f"#{val}" if val else f"ID:{tid}" for tid, val in zip(detections.tracker_id, valid_ocrs)]

    def _update_homography(self, kp_detections: sv.Detections, width: int):
        mask_kp = np.isin([self.detector.kp_model.names[cid] for cid in kp_detections.class_id], config.TARGET_LANDMARKS)
        landmarks = kp_detections[mask_kp]
        
        if len(landmarks) >= 4:
            assigned = map_generic_to_specific_points(landmarks, self.detector.kp_model.names, width)
            src, dst = [], []
            for lbl, pt in assigned.items():
                if lbl in config.RINK_DEFAULT_2D:
                    src.append(pt)
                    dst.append(config.RINK_DEFAULT_2D[lbl])
                    
            if len(src) >= 4:
                new_transformer = ViewTransformer(np.array(src), np.array(dst))
                try:
                    # Test pour vérifier que l'homographie est valide
                    _ = new_transformer.transform_points(points=np.array([[0.0, 0.0]]))
                    return new_transformer
                except Exception:
                    pass
        return None

    def _draw_minimap_overlay(self, frame, base_rink, transformer, players, puck):
        if not transformer:
            return
            
        rink_2d = base_rink.copy()
        if len(players) > 0:
            pts = transformer.transform_points(players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
            rink_2d = draw_points_on_rink(rink_2d, pts, (255,0,0))
            
        if len(puck) > 0:
            pts = transformer.transform_points(puck.get_anchors_coordinates(sv.Position.CENTER))
            rink_2d = draw_points_on_rink(rink_2d, pts, (0,0,0), radius=6)

        rH, rW = rink_2d.shape[:2]
        y, x = frame.shape[0] - rH - 20, frame.shape[1] - rW - 20
        if y > 0 and x > 0:
            cv2.addWeighted(rink_2d, 0.8, frame[y:y+rH, x:x+rW], 0.2, 0, frame[y:y+rH, x:x+rW])
