"""
Model loader module for loading pre-trained CNN models and making predictions.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm
import json
from pathlib import Path
from typing import Tuple, Dict
import numpy as np

from .config import (
    MODEL_NAME, DEVICE, IMAGE_SIZE, 
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
)


class ModelLoader:
    """Handles model loading, preprocessing, and inference."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the model loader.
        
        Args:
            model_name: Name of the model to load (resnet50, resnet152, convnext_large)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = self._load_model()
        self.model.eval()
        
        self.preprocess = self._get_preprocessing()
        self.class_labels = self._load_imagenet_labels()
        
    def _load_model(self) -> nn.Module:
        """Load the pre-trained model."""
        print(f"Loading model: {self.model_name}")
        
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif self.model_name == "resnet152":
            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        elif self.model_name == "convnext_large":
            # Use timm for ConvNeXt
            model = timm.create_model('convnext_large', pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        model = model.to(self.device)
        return model
    
    def _get_preprocessing(self) -> transforms.Compose:
        """Get the preprocessing pipeline for the model."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def _load_imagenet_labels(self) -> Dict[int, str]:
        """Load ImageNet class labels."""
        # We'll create a simple label mapping
        # In production, load from a JSON file
        labels = {}
        try:
            # Try to load from file if it exists
            label_file = Path(__file__).parent.parent / "assets" / "imagenet_labels.json"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = json.load(f)
                    # Convert string keys to int
                    labels = {int(k): v for k, v in labels.items()}
        except Exception as e:
            print(f"Could not load labels: {e}")
            # Provide fallback
            labels = {i: f"class_{i}" for i in range(NUM_CLASSES)}
        
        return labels
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.preprocess(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image: Image.Image, top_k: int = 5) -> Dict:
        """
        Make a prediction on an image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions, probabilities, and class names
        """
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class_id': idx.item(),
                'class_name': self.class_labels.get(idx.item(), f"class_{idx.item()}"),
                'probability': prob.item()
            })
        
        return {
            'top_prediction': predictions[0],
            'top_k_predictions': predictions,
            'raw_output': output.cpu().numpy().tolist()
        }
    
    def get_target_layer(self):
        """Get the target layer for Grad-CAM based on model architecture."""
        if "resnet" in self.model_name:
            return self.model.layer4[-1]
        elif "convnext" in self.model_name:
            # For ConvNeXt, use the last stage
            return self.model.stages[-1]
        else:
            # Default: try to find last convolutional layer
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Conv2d):
                    return module
            raise ValueError("Could not find suitable target layer")


# Global model instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get or create the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def load_model() -> ModelLoader:
    """Load and return the model loader."""
    return get_model_loader()


def predict(image: Image.Image, top_k: int = 5) -> Dict:
    """
    Make a prediction on an image.
    
    Args:
        image: PIL Image
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with predictions
    """
    loader = get_model_loader()
    return loader.predict(image, top_k)
