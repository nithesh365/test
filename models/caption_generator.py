import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time
from config import config
from utils.style_rewriter import get_style_rewriter

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
        Generate caption for an image with style using BLIP + Gemini
        
        Args:
            image: PIL Image object
            style: Caption style (factual, poetic, humorous, cinematic, simple)
        
        Returns:
            dict with caption, confidence, and processing_time
        """
        start_time = time.time()
        
        # Step 1: Generate base caption with BLIP (no style prompt)
        blip_start = time.time()
        inputs = self.processor(
            images=image,
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
        
        # Decode base caption
        base_caption = self.processor.decode(
            output.sequences[0],
            skip_special_tokens=True
        )
        
        # Calculate confidence
        if hasattr(output, 'sequences_scores'):
            confidence = float(torch.exp(output.sequences_scores[0]))
        else:
            scores = torch.stack(output.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            confidence = float(probs.max(dim=-1).values.mean())
        
        blip_time = round(time.time() - blip_start, 3)
        
        # Step 2: Rewrite caption with Gemini for different styles
        gemini_time = 0
        final_caption = base_caption
        
        # Only use Gemini if style is not factual
        if style != "factual":
            try:
                rewriter = get_style_rewriter()
                final_caption, gemini_time = rewriter.rewrite_caption(base_caption, style)
            except Exception as e:
                print(f"Style rewriting failed, using base caption: {e}")
                final_caption = base_caption
        
        total_time = round(time.time() - start_time, 3)
        
        return {
            "caption": final_caption,
            "base_caption": base_caption,  # Include for comparison
            "confidence": round(confidence, 4),
            "processing_time": total_time,
            "blip_time": blip_time,
            "gemini_time": gemini_time,
            "style": style
        }

# Global model instance
caption_model = None

def get_caption_model():
    global caption_model
    if caption_model is None:
        caption_model = CaptionGenerator()
    return caption_model
