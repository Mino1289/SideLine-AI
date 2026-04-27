import cv2
import numpy as np

class ViewTransformer:
    """Calcul d'homographie et projection 2D sur la patinoire."""
    
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m, _ = cv2.findHomography(source.astype(np.float32), target.astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=5.0)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if self.m is None or len(points) == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed.reshape(-1, 2)

def map_generic_to_specific_points(detections, model_names, img_width: int):
    """Heuristique spatiale: assigne 'faceoff' générique au point précis 2D approprié."""
    centroids, goals, faceoffs = [], [], []
    for xyxy, cid in zip(detections.xyxy, detections.class_id):
        name = model_names[cid].lower()
        cx, cy = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0
        
        if "centri" in name: 
            centroids.append((cx, cy))
        elif "goal" in name: 
            goals.append((cx, cy))
        elif "faceoff" in name: 
            faceoffs.append((cx, cy))
        
    assigned_pts = {}
    center_x = centroids[0][0] if centroids else img_width / 2.0
    
    if centroids: 
        assigned_pts["Center Ice"] = centroids[0]
    
    for gx, gy in goals:
        label = "Goal Frame Left" if gx < center_x else "Goal Frame Right"
        assigned_pts[label] = (gx, gy)
        
    left_fs = sorted([p for p in faceoffs if p[0] < center_x], key=lambda x: x[1])
    right_fs = sorted([p for p in faceoffs if p[0] >= center_x], key=lambda x: x[1])

    has_l_goal = "Goal Frame Left" in assigned_pts
    has_r_goal = "Goal Frame Right" in assigned_pts
    
    for i, p in enumerate(left_fs[:2]):
        lbl = "Faceoff Dot Def Left Top" if has_l_goal and i==0 else \
              "Faceoff Dot Def Left Bot" if has_l_goal and i==1 else \
              "Faceoff Dot N Z1" if i==0 else "Faceoff Dot N Z2"
        assigned_pts[lbl] = p
        
    for i, p in enumerate(right_fs[:2]):
        lbl = "Faceoff Dot Def Right Top" if has_r_goal and i==0 else \
              "Faceoff Dot Def Right Bot" if has_r_goal and i==1 else \
              "Faceoff Dot N Z3" if i==0 else "Faceoff Dot N Z4"
        assigned_pts[lbl] = p
        
    return assigned_pts
