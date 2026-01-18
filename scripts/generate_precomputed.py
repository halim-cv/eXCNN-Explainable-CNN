"""
Script to generate precomputed XAI results for all sample images.
Saves results to assets/precomputed/ for the gallery.
"""

import sys
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.model_loader import load_model, predict
from backend.explainers import (
    get_gradcam,
    get_occlusion_map,
    get_guided_backprop,
    get_guided_gradcam
)
from backend.config import PRECOMPUTED_DIR, EXAMPLES_DIR

def save_image(path: Path, image_array: np.ndarray):
    """Save numpy array as image."""
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(path), image_array)

def generate_precomputed():
    """Generate precomputed results for all sample images."""
    print(f"Scanning for images in {EXAMPLES_DIR}...")
    
    # Get all images
    image_files = list(EXAMPLES_DIR.glob("*.jpg")) + list(EXAMPLES_DIR.glob("*.png"))
    
    if not image_files:
        print("No images found! Run prepare_samples.py first.")
        return
    
    print(f"Found {len(image_files)} images. Generating explanations...")
    load_model()  # Preload model
    
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_files):
        try:
            print(f"\nProcessing {img_path.name}...")
            
            # Create output directory for this image
            img_stem = img_path.stem
            out_dir = PRECOMPUTED_DIR / img_stem
            out_dir.mkdir(exist_ok=True)
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # 1. Prediction
            print("  Running prediction...")
            prediction = predict(image)
            predicted_class = prediction['top_prediction']['class_id']
            
            # Save metadata
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(prediction, f, indent=2)
            
            # Save original (resized)
            image_resized = image.resize((224, 224))
            image_resized.save(out_dir / "original.jpg")
            
            # 2. Grad-CAM
            print("  Generating Grad-CAM...")
            cam_vis, cam_heatmap = get_gradcam(image, target_class=predicted_class)
            save_image(out_dir / "gradcam_vis.jpg", cam_vis)
            
            # Save heatmap colored
            heatmap_colored = cv2.applyColorMap(
                (cam_heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            save_image(out_dir / "gradcam_heatmap.jpg", heatmap_colored)
            
            # 3. Occlusion Sensitivity
            print("  Generating Occlusion Sensitivity...")
            # Use larger window/stride for faster precomputation, or detailed for quality
            # We'll use defaults from config but ensure they are reasonable
            occ_vis, occ_map = get_occlusion_map(image, target_class=predicted_class)
            save_image(out_dir / "occlusion_vis.jpg", occ_vis)
            
            occ_heatmap = cv2.applyColorMap(
                (occ_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            occ_heatmap = cv2.cvtColor(occ_heatmap, cv2.COLOR_BGR2RGB)
            save_image(out_dir / "occlusion_heatmap.jpg", occ_heatmap)
            
            # 4. Guided Backprop
            print("  Generating Guided Backprop...")
            gb_map = get_guided_backprop(image, target_class=predicted_class)
            save_image(out_dir / "guided_backprop.jpg", gb_map)
            
            # 5. Guided Grad-CAM
            print("  Generating Guided Grad-CAM...")
            ggc_map = get_guided_gradcam(image, target_class=predicted_class)
            save_image(out_dir / "guided_gradcam.jpg", ggc_map)
            
            print(f"✓ Results saved to {out_dir}")
            
        except Exception as e:
            print(f"✗ Failed to process {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_precomputed()
