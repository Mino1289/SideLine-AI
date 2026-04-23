from pathlib import Path
from typing import Dict

# --- Chemins ---
# ROOT_DIR pointe vers le dossier actuel (hockey_ai_project)
ROOT_DIR = Path(__file__).resolve().parent 
MODELS_DIR = ROOT_DIR / "models"
VIDEO_DIR = ROOT_DIR / "video"

# Chemins des modèles
YOLO_SEG_MODEL_PATH = MODELS_DIR / "yolo26l-seg.pt"
YOLO_SEG_MODEL_PATH_FALLBACK = MODELS_DIR / "yolo26n-seg.pt"
OV_SEG_MODEL_PATH = MODELS_DIR / "yolo26l-seg_int8_openvino_model"

YOLO_KP_MODEL_PATH = MODELS_DIR / "HockeyAI_model_weight.pt"
OV_KEYPOINT_MODEL_PATH = MODELS_DIR / "HockeyAI_model_weight_openvino_model"

# Chemins d'E/S vidéo (à adapter)
SOURCE_VIDEO_PATH = VIDEO_DIR / "hockey" / "hockey_Maple_Leafs_vs_Canadiens_10s.mp4"
OUT_VIDEO_DIR = VIDEO_DIR / "hockey" / "out"

# --- Constantes Patinoire ---
RINK_DEFAULT_2D = {
    "Center Ice": (100.0, 42.5),
    "Goal Frame Left": (11.0, 42.5),   
    "Goal Frame Right": (189.0, 42.5), 
    "Faceoff Dot N Z1": (76.0, 20.5),  
    "Faceoff Dot N Z2": (76.0, 64.5),  
    "Faceoff Dot N Z3": (124.0, 20.5), 
    "Faceoff Dot N Z4": (124.0, 64.5), 
    "Faceoff Dot Def Left Top": (31.0, 20.5),
    "Faceoff Dot Def Left Bot": (31.0, 64.5),
    "Faceoff Dot Def Right Top": (169.0, 20.5),
    "Faceoff Dot Def Right Bot": (169.0, 64.5),
}
TARGET_LANDMARKS = ["centriod", "centroid", "faceoff", "goal"]

# --- Équipes ---
TEAM_COLORS = {
    "Montreal Canadiens": "#AF1E2D",
    "Toronto Maple Leafs": "#9DC1F0"
}

TEAM_ROSTERS: Dict[str, Dict[str, str]] = {
  "Montreal Canadiens": {
    "8": "Matheson", "11": "Gallagher", "13": "Caufield", "14": "Suzuki", "15": "Newhook",
    "17": "Anderson", "20": "Slafkovsky", "21": "Guhle", "24": "Danault", "28": "Dvorak",
    "35": "Montembeault", "40": "Armia", "42": "Engstrom", "45": "Carrier", "47": "Struble",
    "48": "Hutson", "51": "Heineman", "53": "Dobson", "55": "Pezzetta", "58": "Savard",
    "71": "Evans", "72": "Xhekaj", "75": "Dobes", "76": "Bolduc", "77": "Dach",
    "85": "Texier", "90": "Veleno", "91": "Kapanen", "92": "Laine", "93": "Demidov"
  },
  "Toronto Maple Leafs": {
    "2": "Benoit", "8": "Tanev", "11": "Domi", "18": "Lorentz", "19": "Jarnkrok",
    "22": "McCabe", "23": "Knies", "25": "Carlo", "26": "Quillan", "28": "Stecher",
    "29": "Groulx", "34": "Matthews", "41": "Stolarz", "43": "Haymes", "44": "Rielly",
    "51": "Myers", "53": "Cowan", "60": "Woll", "61": "Pezzetta", "63": "Maccelli",
    "70": "Akhtyamov", "76": "Villeneuve", "77": "Tverberg", "81": "Joshua", "88": "Nylander",
    "89": "Robertson", "91": "Tavares", "95": "Ekman-Larsson"
  }
}
