import logging
from datetime import datetime
import config
from core.pipeline import HockeyVideoPipeline

# Configuration propre du logger (au lieu de print)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # S"assurer que le dossier de sortie existe
    config.OUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_video = config.OUT_VIDEO_DIR / f"output_hockey_{timestamp}.mp4"

    # Instanciation et lancement
    pipeline = HockeyVideoPipeline(
        source_path=config.SOURCE_VIDEO_PATH,
        target_path=target_video
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()
