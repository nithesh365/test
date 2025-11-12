import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time
from config import config

class CaptionGenerator:
    def __init__(self):
        print(f"Loading model on {config.DEVICE}...")
        self.processor = BlipProcessor.from_pretrained(config.MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(
            config.MODEL_NAME
        ).to(config.DEVICE)
        print("Model loaded successfully!")
    
    def generate_caption(self, image: Image.Image, style: str = "factual"):
        """
        Generate caption for an image with optional style
        
        Args:
            image: PIL Image object
            style: Caption style (factual, poetic, humorous, cinematic, simple)
        
        Returns:
            dict with caption, confidence, and processing_time
        """
        start_time = time.time()
        
        # Get style prompt
        text_prompt = config.STYLE_PROMPTS.get(style, config.STYLE_PROMPTS["factual"])
        
        # Process image
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(config.DEVICE)
        
        # Generate caption
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=config.MAX_LENGTH,
                num_beams=config.NUM_BEAMS,
                temperature=0.7,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode caption
        caption = self.processor.decode(
            output.sequences[0],
            skip_special_tokens=True
        )
        
        # Calculate confidence (average of token scores)
        if hasattr(output, 'sequences_scores'):
            confidence = float(torch.exp(output.sequences_scores[0]))
        else:
            # Fallback confidence calculation
            scores = torch.stack(output.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            confidence = float(probs.max(dim=-1).values.mean())
        
        processing_time = round(time.time() - start_time, 3)
        
        return {
            "caption": caption,
            "confidence": round(confidence, 4),
            "processing_time": processing_time,
            "style": style
        }

# Global model instance
caption_model = None

def get_caption_model():
    global caption_model
    if caption_model is None:
        caption_model = CaptionGenerator()
    return caption_model
