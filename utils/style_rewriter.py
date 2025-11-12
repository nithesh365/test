# pip install groq

import os
from groq import Groq
from config import config
import time

class StyleRewriter:
    def __init__(self):
        """Initialize Groq API"""
        if not hasattr(config, 'GROQ_API_KEY') or not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=config.GROQ_API_KEY)
        print("Groq API initialized successfully!")
    
    def rewrite_caption(self, base_caption: str, style: str) -> tuple:
        start_time = time.time()
        
        if not base_caption or not base_caption.strip():
            return base_caption, 0.0
        
        if not style or not style.strip():
            style = "creative"
        
        prompt = f"Rewrite this image caption in a {style} style. Give ONLY the rewritten caption, nothing else.\n\nOriginal: {base_caption}\n\nRewritten:"
        
        rewritten = base_caption
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",  # Fast and high quality
                temperature=0.7,
                max_tokens=150,
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            # Clean up
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
            print(f"✅ Groq: {rewritten[:80]}...")
            
        except Exception as e:
            print(f"❌ Groq error: {e}")
        
        processing_time = round(time.time() - start_time, 3)
        return rewritten, processing_time

style_rewriter = None

def get_style_rewriter():
    global style_rewriter
    if style_rewriter is None:
        style_rewriter = StyleRewriter()
    return style_rewriter
