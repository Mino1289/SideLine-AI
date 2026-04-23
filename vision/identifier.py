import easyocr
import numpy as np
from typing import List, Optional

class OCRIdentifier:
    """Classe responsable de la lecture de numéros de maillots par OCR."""
    
    def __init__(self, gpu: bool = True):
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    def read_number(self, image_crop: np.ndarray) -> Optional[str]:
        if image_crop.size == 0:
            return None
        result = self.reader.readtext(image_crop, allowlist='0123456789', detail=0)
        return result[0] if result else None

class OCRHistoryTracker:
    """Valide les lectures OCR via récurrence temporelle (filtre de bruits)."""
    
    def __init__(self, n_consecutive: int = 3):
        self.n_consecutive = n_consecutive
        self.history = {}
        self.validated = {}
        
    def update(self, tracker_ids: List[int], values: List[Optional[str]]):
        for tid, val in zip(tracker_ids, values):
            if not val:
                continue
            if tid not in self.history:
                self.history[tid] = []
            
            self.history[tid].append(val)
            self.history[tid] = self.history[tid][-self.n_consecutive:]
            
            if len(self.history[tid]) == self.n_consecutive and len(set(self.history[tid])) == 1:
                self.validated[tid] = val
                
    def get_validated(self, tracker_ids: List[int]) -> List[Optional[str]]:
        return [self.validated.get(tid, None) for tid in tracker_ids]
