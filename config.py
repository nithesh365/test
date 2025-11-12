import os
from pathlib import Path

class Config:
    # Model settings
    MODEL_NAME = "Salesforce/blip-image-captioning-large"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # API settings
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Caption generation settings
    MAX_LENGTH = 50
    NUM_BEAMS = 5
    
    # Style prompts
    STYLE_PROMPTS = {
        "factual": "a photography of",
        "poetic": "an artistic depiction of",
        "humorous": "a funny image showing",
        "cinematic": "a dramatic scene featuring",
        "simple": "this is"
    }
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000

config = Config()
