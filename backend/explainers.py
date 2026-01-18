"""
XAI (Explainable AI) module implementing multiple explanation methods.
Includes Grad-CAM, Occlusion Sensitivity, Guided Backpropagation, and Guided Grad-CAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .model_loader import get_model_loader
from .config import OCCLUSION_WINDOW_SIZE, OCCLUSION_STRIDE, OCCLUSION_BASELINE


class Explainer:
    """Handles multiple XAI explanation methods."""
    
    def __init__(self):
        """Initialize the explainer with the loaded model."""
        self.model_loader = get_model_loader()
        self.model = self.model_loader.model
        self.device = self.model_loader.device
        
    def get_gradcam(
        self, 
        image: Image.Image, 
        target_class: Optional[int] = None,
        use_rgb: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM explanation.
        
        Args:
            image: Input PIL Image
            target_class: Target class for explanation (None = predicted class)
            use_rgb: Whether to return RGB overlay or just heatmap
            
        Returns:
            Tuple of (cam_image, heatmap)
        """
        # Get target layer
        target_layers = [self.model_loader.get_target_layer()]
        
        # Create Grad-CAM object
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        # Preprocess image
        input_tensor = self.model_loader.preprocess_image(image)
        
        # If no target class specified, use the predicted class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first image
        
        # Convert original image to RGB array
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        
        # Create visualization
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=use_rgb)
        
        return cam_image, grayscale_cam
    
    def get_occlusion_map(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
        window_size: int = OCCLUSION_WINDOW_SIZE,
        stride: int = OCCLUSION_STRIDE,
        baseline: str = OCCLUSION_BASELINE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate occlusion sensitivity map.
        
        Args:
            image: Input PIL Image
            target_class: Target class for explanation (None = predicted class)
            window_size: Size of occlusion window
            stride: Stride for sliding window
            baseline: Baseline for occlusion ('mean', 'zero', 'blur')
            
        Returns:
            Tuple of (visualization, sensitivity_map)
        """
        # Preprocess image
        input_tensor = self.model_loader.preprocess_image(image)
        original_image = np.array(image.resize((224, 224)))
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Get baseline prediction score
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            baseline_score = F.softmax(baseline_output, dim=1)[0, target_class].item()
        
        # Initialize sensitivity map
        img_size = 224
        sensitivity_map = np.zeros((img_size, img_size))
        counts = np.zeros((img_size, img_size))
        
        # Create occlusion baseline
        if baseline == "mean":
            occlusion_value = input_tensor.mean()
        elif baseline == "zero":
            occlusion_value = 0.0
        else:  # blur
            occlusion_value = None  # We'll handle blur separately
        
        # Slide window across image
        for i in range(0, img_size - window_size + 1, stride):
            for j in range(0, img_size - window_size + 1, stride):
                # Create occluded image
                occluded_tensor = input_tensor.clone()
                
                if baseline == "blur":
                    # Apply Gaussian blur to the region
                    occluded_np = occluded_tensor[0].cpu().numpy().transpose(1, 2, 0)
                    blurred = cv2.GaussianBlur(occluded_np, (window_size, window_size), 0)
                    blurred[i:i+window_size, j:j+window_size] = cv2.GaussianBlur(
                        occluded_np[i:i+window_size, j:j+window_size],
                        (window_size, window_size),
                        10
                    )
                    occluded_tensor = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                else:
                    occluded_tensor[0, :, i:i+window_size, j:j+window_size] = occlusion_value
                
                # Get prediction with occlusion
                with torch.no_grad():
                    occluded_output = self.model(occluded_tensor)
                    occluded_score = F.softmax(occluded_output, dim=1)[0, target_class].item()
                
                # Calculate sensitivity (drop in probability)
                sensitivity = baseline_score - occluded_score
                
                # Record sensitivity
                sensitivity_map[i:i+window_size, j:j+window_size] += sensitivity
                counts[i:i+window_size, j:j+window_size] += 1
        
        # Average overlapping regions
        sensitivity_map = np.divide(
            sensitivity_map, 
            counts, 
            where=counts != 0
        )
        
        # Normalize to [0, 1]
        if sensitivity_map.max() > 0:
            sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min())
        
        # Create heatmap visualization
        heatmap = cv2.applyColorMap(
            (sensitivity_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
        
        return overlay, sensitivity_map
    
    def get_guided_backprop(
        self,
        image: Image.Image,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Guided Backpropagation saliency map.
        
        Args:
            image: Input PIL Image
            target_class: Target class for explanation (None = predicted class)
            
        Returns:
            Guided backprop saliency map
        """
        # Preprocess image
        input_tensor = self.model_loader.preprocess_image(image)
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Use pytorch_grad_cam's GuidedBackpropReLUModel
        guided_model = GuidedBackpropReLUModel(model=self.model, device=self.device)
        
        # Generate guided backprop
        guided_grads = guided_model(
            input_tensor,
            target_category=target_class
        )
        
        if len(guided_grads.shape) == 4:
            # Check if it's (N, C, H, W)
            guided_grads = guided_grads[0]
            
        # Check shape before transpose
        if len(guided_grads.shape) == 3:
            if guided_grads.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                guided_grads = guided_grads.transpose(1, 2, 0)
            # else assumed already (H, W, C)
        
        # Normalize
        guided_grads = (guided_grads - guided_grads.min()) / (guided_grads.max() - guided_grads.min() + 1e-8)
        guided_grads = (guided_grads * 255).astype(np.uint8)
        
        return guided_grads
    
    def get_guided_gradcam(
        self,
        image: Image.Image,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Guided Grad-CAM (combines Grad-CAM and Guided Backprop).
        
        Args:
            image: Input PIL Image
            target_class: Target class for explanation (None = predicted class)
            
        Returns:
            Guided Grad-CAM visualization
        """
        # Get Grad-CAM heatmap
        _, gradcam_mask = self.get_gradcam(image, target_class, use_rgb=False)
        
        # Get Guided Backprop
        guided_backprop = self.get_guided_backprop(image, target_class)
        
        # Resize gradcam to match guided backprop
        gradcam_resized = cv2.resize(gradcam_mask, (guided_backprop.shape[1], guided_backprop.shape[0]))
        
        # Expand to 3 channels
        gradcam_3d = np.repeat(gradcam_resized[:, :, np.newaxis], 3, axis=2)
        
        # Element-wise multiplication
        guided_gradcam = guided_backprop * gradcam_3d
        
        # Normalize
        guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min() + 1e-8)
        guided_gradcam = (guided_gradcam * 255).astype(np.uint8)
        
        return guided_gradcam


# Global explainer instance
_explainer = None


def get_explainer() -> Explainer:
    """Get or create the global explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = Explainer()
    return _explainer


# Convenience functions
def get_gradcam(image: Image.Image, target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM explanation."""
    return get_explainer().get_gradcam(image, target_class)


def get_occlusion_map(
    image: Image.Image, 
    target_class: Optional[int] = None,
    window_size: int = OCCLUSION_WINDOW_SIZE,
    stride: int = OCCLUSION_STRIDE,
    baseline: str = OCCLUSION_BASELINE
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate occlusion sensitivity map."""
    return get_explainer().get_occlusion_map(
        image, target_class, window_size, stride, baseline
    )


def get_guided_backprop(image: Image.Image, target_class: Optional[int] = None) -> np.ndarray:
    """Generate guided backpropagation map."""
    return get_explainer().get_guided_backprop(image, target_class)


def get_guided_gradcam(image: Image.Image, target_class: Optional[int] = None) -> np.ndarray:
    """Generate guided Grad-CAM map."""
    return get_explainer().get_guided_gradcam(image, target_class)
