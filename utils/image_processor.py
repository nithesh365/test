from PIL import Image
import io
from config import config

class ImageProcessor:
    @staticmethod
    def validate_image(file_content: bytes) -> bool:
        """Validate image size and format"""
        if len(file_content) > config.MAX_IMAGE_SIZE:
            raise ValueError(f"Image size exceeds {config.MAX_IMAGE_SIZE / (1024*1024)}MB")
        
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()
            return True
        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}")
    
    @staticmethod
    def load_image(file_content: bytes) -> Image.Image:
        """Load and preprocess image"""
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if too large (keep aspect ratio)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return image
