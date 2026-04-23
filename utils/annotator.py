import cv2
import numpy as np

def draw_hockey_rink_2d(scale: int = 3) -> np.ndarray:
    """Dessine la patinoire de référence 200x85 pi."""
    w, h = int(200 * scale), int(85 * scale)
    rink = np.ones((h, w, 3), dtype=np.uint8) * 250
    
    cv2.line(rink, (int(100*scale), 0), (int(100*scale), h), (0, 0, 255), 2)
    cv2.line(rink, (int(75*scale), 0), (int(75*scale), h), (255, 0, 0), 2)
    cv2.line(rink, (int(125*scale), 0), (int(125*scale), h), (255, 0, 0), 2)
    cv2.line(rink, (int(11*scale), 0), (int(11*scale), h), (0, 0, 255), 1)
    cv2.line(rink, (int(189*scale), 0), (int(189*scale), h), (0, 0, 255), 1)
    
    cv2.circle(rink, (int(100*scale), int(42.5*scale)), int(15*scale), (255, 0, 0), 2)
    
    for x in [31, 169]:
        for y in [20.5, 64.5]:
            cv2.circle(rink, (int(x*scale), int(y*scale)), int(15*scale), (0, 0, 255), 1)
            cv2.circle(rink, (int(x*scale), int(y*scale)), int(1*scale), (0, 0, 255), -1)
            
    for x in [76, 124]:
        for y in [20.5, 64.5]:
            cv2.circle(rink, (int(x*scale), int(y*scale)), int(1*scale), (0, 0, 255), -1)
            
    cv2.rectangle(rink, (0, 0), (w, h), (0, 0, 0), 3)
    return rink

def draw_points_on_rink(rink_img: np.ndarray, points: np.ndarray, color: tuple, scale: int = 3, radius: int = 4) -> np.ndarray:
    """Dessine les points projetés sur la mini-carte de la patinoire."""
    img = rink_img.copy()
    if points is None or len(points) == 0:
        return img
        
    for pt in points:
        x, y = int(pt[0] * scale), int(pt[1] * scale)
        x = max(0, min(x, img.shape[1]-1))
        y = max(0, min(y, img.shape[0]-1))
        cv2.circle(img, (x, y), radius, color, -1)
    return img
