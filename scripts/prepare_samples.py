"""
Script to download and prepare sample ImageNet images for the example gallery.
Downloads a small set of representative images instead of the full dataset.
"""

import requests
from pathlib import Path
import json
from PIL import Image
import io

# Sample ImageNet classes with representative images
SAMPLE_CLASSES = {
    "281": {"name": "tabby_cat", "label": "tabby, tabby cat"},
    "207": {"name": "golden_retriever", "label": "golden retriever"},
    "954": {"name": "banana", "label": "banana"},
    "504": {"name": "coffee_mug", "label": "coffee mug"},
    "859": {"name": "toaster", "label": "toaster"},
    "530": {"name": "dining_table", "label": "dining table, board"},
    "717": {"name": "pickup_truck", "label": "pickup, pickup truck"},
    "340": {"name": "zebra", "label": "zebra"},
    "388": {"name": "giant_panda", "label": "giant panda, panda"},
    "701": {"name": "parachute", "label": "parachute, chute"}
}

# URLs for sample images (using Unsplash or similar free image sources)
# Note: In production, you'd download actual ImageNet images or use your own
SAMPLE_IMAGE_URLS = {
    "281": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",  # Cat
    "207": "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400",  # Golden Retriever
    "954": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=400",  # Banana
    "504": "https://images.unsplash.com/photo-1514481538271-cf9f99627ab4?w=400",  # Coffee Mug
    "859": "https://images.unsplash.com/photo-1585659722983-3a675dabf23d?w=400",  # Toaster
}


def download_sample_images(output_dir: Path, num_samples: int = 5):
    """
    Download a small set of sample images for demonstration.
    
    Args:
        output_dir: Directory to save images
        num_samples: Number of images to download
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_samples} sample images...")
    
    downloaded = 0
    for class_id, url in list(SAMPLE_IMAGE_URLS.items())[:num_samples]:
        class_info = SAMPLE_CLASSES[class_id]
        filename = f"{class_id}_{class_info['name']}.jpg"
        filepath = output_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists")
            downloaded += 1
            continue
        
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Open and save image
            image = Image.open(io.BytesIO(response.content))
            image = image.convert('RGB')
            image.save(filepath, 'JPEG', quality=95)
            
            print(f"✓ Downloaded {filename}")
            downloaded += 1
            
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    print(f"\nDownloaded {downloaded}/{num_samples} images")
    return downloaded


def create_placeholder_images(output_dir: Path, num_samples: int = 5):
    """
    Create placeholder images if download fails.
    
    Args:
        output_dir: Directory to save images
        num_samples: Number of placeholder images to create
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} placeholder images...")
    
    from PIL import ImageDraw, ImageFont
    
    for i, (class_id, class_info) in enumerate(list(SAMPLE_CLASSES.items())[:num_samples]):
        filename = f"{class_id}_{class_info['name']}.jpg"
        filepath = output_dir / filename
        
        if filepath.exists():
            continue
        
        # Create a colored placeholder
        img = Image.new('RGB', (400, 400), color=(73 + i*20, 109 + i*15, 137 + i*10))
        draw = ImageDraw.Draw(img)
        
        # Add text
        text = class_info['label']
        # Use default font
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((400 - text_width) // 2, (400 - text_height) // 2)
        
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        img.save(filepath, 'JPEG', quality=95)
        print(f"✓ Created placeholder for {filename}")


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "assets" / "examples_imagenet"
    
    # Try to download real images
    num_downloaded = download_sample_images(examples_dir, num_samples=5)
    
    # Create placeholders for any missing images
    if num_downloaded < 5:
        print("\nCreating placeholders for missing images...")
        create_placeholder_images(examples_dir, num_samples=5)
    
    print(f"\n✓ Sample images ready in {examples_dir}")
