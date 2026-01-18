"""
Test script to verify the backend functionality.
Run this after installing dependencies to ensure everything works.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image

def test_imports():
    """Test that all required packages are importable."""
    print("Testing imports...")
    
    try:
        import torch
        import torchvision
        import timm
        from pytorch_grad_cam import GradCAM
        import cv2
        from fastapi import FastAPI
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_loading():
    """Test that the model loads correctly."""
    print("\nTesting model loading...")
    
    try:
        from backend.model_loader import load_model, get_model_loader
        
        loader = load_model()
        print(f"✓ Model loaded: {loader.model_name}")
        print(f"✓ Device: {loader.device}")
        print(f"✓ Number of classes: {len(loader.class_labels)}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_prediction():
    """Test prediction on a dummy image."""
    print("\nTesting prediction...")
    
    try:
        from backend.model_loader import predict
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Make prediction
        result = predict(dummy_image, top_k=5)
        
        print(f"✓ Prediction successful")
        print(f"  Top prediction: {result['top_prediction']['class_name']}")
        print(f"  Confidence: {result['top_prediction']['probability']:.4f}")
        print(f"  Top 5 classes: {[p['class_name'] for p in result['top_k_predictions']]}")
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradcam():
    """Test Grad-CAM explanation."""
    print("\nTesting Grad-CAM...")
    
    try:
        from backend.explainers import get_gradcam
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Generate Grad-CAM
        cam_image, heatmap = get_gradcam(dummy_image, target_class=None)
        
        print(f"✓ Grad-CAM successful")
        print(f"  CAM image shape: {cam_image.shape}")
        print(f"  Heatmap shape: {heatmap.shape}")
        return True
    except Exception as e:
        print(f"✗ Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_occlusion():
    """Test Occlusion Sensitivity (with small settings for speed)."""
    print("\nTesting Occlusion Sensitivity (this may take a moment)...")
    
    try:
        from backend.explainers import get_occlusion_map
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Generate occlusion map with small window for faster testing
        overlay, sensitivity_map = get_occlusion_map(
            dummy_image, 
            target_class=None,
            window_size=20,  # Larger window for faster test
            stride=20  # Larger stride for faster test
        )
        
        print(f"✓ Occlusion Sensitivity successful")
        print(f"  Overlay shape: {overlay.shape}")
        print(f"  Sensitivity map shape: {sensitivity_map.shape}")
        return True
    except Exception as e:
        print(f"✗ Occlusion Sensitivity failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_guided_backprop():
    """Test Guided Backpropagation."""
    print("\nTesting Guided Backpropagation...")
    
    try:
        from backend.explainers import get_guided_backprop
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Generate guided backprop
        saliency_map = get_guided_backprop(dummy_image, target_class=None)
        
        print(f"✓ Guided Backpropagation successful")
        print(f"  Saliency map shape: {saliency_map.shape}")
        return True
    except Exception as e:
        print(f"✗ Guided Backpropagation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_guided_gradcam():
    """Test Guided Grad-CAM."""
    print("\nTesting Guided Grad-CAM...")
    
    try:
        from backend.explainers import get_guided_gradcam
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Generate guided grad-cam
        guided_gradcam = get_guided_gradcam(dummy_image, target_class=None)
        
        print(f"✓ Guided Grad-CAM successful")
        print(f"  Guided Grad-CAM shape: {guided_gradcam.shape}")
        return True
    except Exception as e:
        print(f"✗ Guided Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("eXCNN Backend Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction),
        ("Grad-CAM", test_gradcam),
        ("Occlusion Sensitivity", test_occlusion),
        ("Guided Backpropagation", test_guided_backprop),
        ("Guided Grad-CAM", test_guided_gradcam),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The backend is ready to use.")
        print("\nNext steps:")
        print("1. Run the API server: uvicorn backend.api:app --reload")
        print("2. Open frontend/index.html in a browser")
        print("3. Try uploading an image!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
