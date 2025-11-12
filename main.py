from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from models.caption_generator import get_caption_model
from config import config
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Image Caption Generator API",
    description="Generate captions for images with multiple style options",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("Starting Image Caption Generator API...")
    get_caption_model()
    print("API ready!")

@app.get("/")
async def root():
    return {
        "message": "Image Caption Generator API",
        "docs": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False
    )
