import google.generativeai as genai
from config import config
import time

class StyleRewriter:
    def __init__(self):
        """Initialize Gemini API"""
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        print("Gemini API initialized successfully!")
    
    def rewrite_caption(self, base_caption: str, style: str) -> tuple:
        """
        Rewrite caption in specified style using Gemini
        
        Args:
            base_caption: Original caption from BLIP
            style: Desired style (factual, poetic, humorous, cinematic, simple)
        
        Returns:
            tuple: (rewritten_caption, processing_time)
        """
        start_time = time.time()
        
        # Get style instruction
        instruction = config.STYLE_INSTRUCTIONS.get(
            style, 
            config.STYLE_INSTRUCTIONS["factual"]
        )
        
        # Create prompt for Gemini
        prompt = f"""{instruction}

Original caption: "{base_caption}"

Rewritten caption:"""
        
        try:
            # Generate with Gemini
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 100,
                }
            )
            
            rewritten = response.text.strip()
            
            # Remove quotes if Gemini added them
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
            
            processing_time = round(time.time() - start_time, 3)
            
            return rewritten, processing_time
        
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            # Fallback to original caption if Gemini fails
            processing_time = round(time.time() - start_time, 3)
            return base_caption, processing_time

# Global instance
style_rewriter = None

def get_style_rewriter():
    global style_rewriter
    if style_rewriter is None:
        style_rewriter = StyleRewriter()
    return style_rewriter
