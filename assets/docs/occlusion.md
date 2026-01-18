# Occlusion Sensitivity

## What is Occlusion Sensitivity?

Occlusion sensitivity is a perturbation-based XAI method that systematically occludes (hides) parts of the input image to measure the impact on the model's prediction. By observing how the prediction changes when different regions are hidden, we can infer which parts of the image are most important for the decision.

## How It Works

1. **Baseline Prediction**: Get the model's prediction on the original image
2. **Sliding Window**: Slide a small window (patch) across the entire image
3. **Occlusion**: For each position, occlude that region (replace with mean/zero/blur)
4. **Re-prediction**: Get the model's prediction with the occluded image
5. **Sensitivity Calculation**: Measure the drop in prediction confidence
6. **Heatmap Creation**: Create a sensitivity map showing importance of each region

## What to Look For

- **High sensitivity regions** (bright/red): Occluding these regions causes large drops in confidence
- **Low sensitivity regions** (dark/blue): Occluding these regions has minimal impact
- **Spatial patterns**: Which parts of the object are most critical

## Advantages

✅ Model-agnostic: Works with any classifier, not just CNNs  
✅ Intuitive: Easy to understand and explain to non-experts  
✅ No gradient required: Works with black-box models  
✅ Robust: Not affected by gradient saturation issues  

## Limitations

⚠️ Computationally expensive: Requires many forward passes  
⚠️ Out-of-distribution: Occluded images may be unrealistic  
⚠️ Window size dependency: Results can vary based on patch size  
⚠️ Time-consuming: Slower than gradient-based methods  

## Parameters

### Window Size
- **Small (5-10px)**: Fine-grained sensitivity, slower computation
- **Medium (15-20px)**: Balance between detail and speed
- **Large (30-50px)**: Coarse regions, faster but less detailed

### Occlusion Type
- **Mean**: Replace with mean pixel value (default)
- **Zero**: Replace with black pixels
- **Blur**: Apply Gaussian blur to the region

## Interpretation Tips

- Larger window sizes give more stable but coarser results
- The method reveals what the model has learned to rely on
- Unexpected high-sensitivity regions may indicate learned biases or shortcuts
- Compare with Grad-CAM to validate findings
