import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    # Model settings
    MODEL_NAME = "Salesforce/blip-image-captioning-large"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    GROQ_API_KEY="apikey"

    
    # API settings
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Caption generation settings
    MAX_LENGTH = 50
    NUM_BEAMS = 5
    
    # Style instructions for Gemini
    STYLE_INSTRUCTIONS = {
        "factual": "Rewrite this image caption as a factual, objective description with specific details. Keep it concise and informative.",
        "poetic": "Rewrite this image caption as beautiful, poetic prose with artistic and expressive language. Make it evocative and emotional.",
        "humorous": "Rewrite this image caption as a funny, witty caption like a meme or social media post. Make it entertaining and clever.",
        "cinematic": "Rewrite this image caption as a dramatic movie scene description with cinematic language. Make it feel epic and storytelling.",
        "simple": "Rewrite this image caption in very simple words that a 5-year-old child would understand. Use basic vocabulary."
    }
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000

config = Config()
