from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from api.schemas import CaptionResponse, ErrorResponse
from models.caption_generator import get_caption_model
from utils.image_processor import ImageProcessor
from config import config

router = APIRouter()

@router.post(
    "/caption/generate",
    response_model=CaptionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def generate_caption(
    image: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)"),
    style: str = Form("factual", description="Caption style: factual, poetic, humorous, cinematic, simple")
):
    """
    Generate a caption for the uploaded image
    
    - **image**: Image file to caption
    - **style**: Desired caption style (optional, defaults to factual)
    """
    
    # Validate style
    if style not in config.STYLE_INSTRUCTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style. Choose from: {list(config.STYLE_PROMPTS.keys())}"
        )
    
    try:
        # Read and validate image
        file_content = await image.read()
        ImageProcessor.validate_image(file_content)
        
        # Load and process image
        pil_image = ImageProcessor.load_image(file_content)
        
        # Generate caption
        model = get_caption_model()
        result = model.generate_caption(pil_image, style)
        
        return CaptionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/health")
async def health_check():
    """Check API health status"""
    return {"status": "healthy", "model_loaded": get_caption_model() is not None}
