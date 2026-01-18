# Grad-CAM: Gradient-weighted Class Activation Mapping

## What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that produces visual explanations for decisions made by convolutional neural networks. It uses the gradients flowing into the final convolutional layer to understand the importance of each neuron for a decision of interest.

## How It Works

1. **Forward Pass**: The image is passed through the network to get predictions
2. **Target Selection**: A target class is selected (usually the predicted class)
3. **Backward Pass**: Gradients of the target class score are computed with respect to feature maps
4. **Weighting**: Feature maps are weighted by the gradients
5. **Combination**: Weighted feature maps are combined and passed through a ReLU
6. **Upsampling**: The resulting heatmap is upsampled to the input image size

## What to Look For

- **Bright/Hot regions** (red, yellow): Areas the model considers important for the prediction
- **Dark/Cool regions** (blue, purple): Areas that don't contribute much to the prediction
- **Focus distribution**: Whether the model focuses on relevant object features or irrelevant background

## Advantages

✅ Class-discriminative: Shows regions specific to the predicted class  
✅ High-level semantic information: Highlights conceptually meaningful regions  
✅ Computationally efficient: Only requires one forward and backward pass  
✅ Model-agnostic: Works with any CNN architecture  

## Limitations

⚠️ Resolution: Limited by the feature map resolution (typically 7x7 or 14x14)  
⚠️ Localization: May not capture fine-grained details  
⚠️ Multiple objects: Can struggle when multiple objects of the same class are present  

## Interpretation Tips

- Look for whether the heatmap highlights the actual object or background elements
- Check if the model focuses on discriminative features (e.g., dog's face vs. random texture)
- Compare heatmaps for different classes to understand what makes them distinct
