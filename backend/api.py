"""
FastAPI backend for the eXCNN application.
Provides endpoints for model prediction and various XAI explanations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import numpy as np
from typing import Optional
import cv2

from .model_loader import get_model_loader, predict
from .explainers import (
    get_gradcam, 
    get_occlusion_map, 
    get_guided_backprop, 
    get_guided_gradcam
)


from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ... (imports)

# Create FastAPI app
app = FastAPI(
    title="eXCNN API",
    description="Explainable CNN Visualization API",
    version="1.0.0"
)

# Mount assets directory
# This allows access to http://localhost:8000/assets/precomputed/...
assets_path = Path(__file__).parent.parent / "assets"
app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy array to base64 encoded string.
    
    Args:
        image_array: numpy array (H, W, 3)
        
    Returns:
        Base64 encoded string
    """
    # Convert to PIL Image
    image = Image.fromarray(image_array.astype(np.uint8))
    
    # Convert to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


def load_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Load image from uploaded file.
    
    Args:
        file: Uploaded file
        
    Returns:
        PIL Image
    """
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Loading models...")
    get_model_loader()
    print("Models loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "eXCNN API - Explainable CNN Visualization",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "gradcam": "/explain/gradcam",
            "occlusion": "/explain/occlusion",
            "guided_backprop": "/explain/guidedbackprop",
            "guided_gradcam": "/explain/guided_gradcam"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), top_k: int = Form(5)):
    """
    Predict class for uploaded image.
    
    Args:
        file: Image file
        top_k: Number of top predictions to return
        
    Returns:
        JSON with predictions
    """
    try:
        image = load_image_from_upload(file)
        result = predict(image, top_k=top_k)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...),
    target_class: Optional[int] = Form(None)
):
    """
    Generate Grad-CAM explanation.
    
    Args:
        file: Image file
        target_class: Target class ID (None = predicted class)
        
    Returns:
        JSON with base64 encoded visualization and heatmap
    """
    try:
        image = load_image_from_upload(file)
        cam_image, heatmap = get_gradcam(image, target_class)
        
        # Convert heatmap to RGB for visualization
        heatmap_rgb = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
        
        return JSONResponse(content={
            "method": "grad-cam",
            "visualization": image_to_base64(cam_image),
            "heatmap": image_to_base64(heatmap_rgb),
            "target_class": target_class
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/occlusion")
async def occlusion_endpoint(
    file: UploadFile = File(...),
    target_class: Optional[int] = Form(None),
    window_size: int = Form(15),
    stride: int = Form(8)
):
    """
    Generate Occlusion Sensitivity map.
    
    Args:
        file: Image file
        target_class: Target class ID (None = predicted class)
        window_size: Size of occlusion window
        stride: Stride for sliding window
        
    Returns:
        JSON with base64 encoded visualization and sensitivity map
    """
    try:
        image = load_image_from_upload(file)
        overlay, sensitivity_map = get_occlusion_map(
            image, 
            target_class,
            window_size=window_size,
            stride=stride
        )
        
        # Convert sensitivity map to RGB
        sensitivity_rgb = cv2.applyColorMap(
            (sensitivity_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        sensitivity_rgb = cv2.cvtColor(sensitivity_rgb, cv2.COLOR_BGR2RGB)
        
        return JSONResponse(content={
            "method": "occlusion-sensitivity",
            "visualization": image_to_base64(overlay),
            "sensitivity_map": image_to_base64(sensitivity_rgb),
            "target_class": target_class,
            "window_size": window_size,
            "stride": stride
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/guidedbackprop")
async def guided_backprop_endpoint(
    file: UploadFile = File(...),
    target_class: Optional[int] = Form(None)
):
    """
    Generate Guided Backpropagation map.
    
    Args:
        file: Image file
        target_class: Target class ID (None = predicted class)
        
    Returns:
        JSON with base64 encoded saliency map
    """
    try:
        image = load_image_from_upload(file)
        saliency_map = get_guided_backprop(image, target_class)
        
        return JSONResponse(content={
            "method": "guided-backprop",
            "visualization": image_to_base64(saliency_map),
            "target_class": target_class
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/guided_gradcam")
async def guided_gradcam_endpoint(
    file: UploadFile = File(...),
    target_class: Optional[int] = Form(None)
):
    """
    Generate Guided Grad-CAM explanation.
    
    Args:
        file: Image file
        target_class: Target class ID (None = predicted class)
        
    Returns:
        JSON with base64 encoded visualization
    """
    try:
        image = load_image_from_upload(file)
        guided_gradcam = get_guided_gradcam(image, target_class)
        
        return JSONResponse(content={
            "method": "guided-gradcam",
            "visualization": image_to_base64(guided_gradcam),
            "target_class": target_class
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
