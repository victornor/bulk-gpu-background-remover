#!/usr/bin/env python3
"""
Bulk background removal script using BiRefNet.
Processes all images in an input directory and saves them with transparent backgrounds.
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def setup_model(device):
    """Load and setup the BiRefNet model."""
    print(f"Loading BiRefNet model on {device}...")
    torch.set_float32_matmul_precision("high")
    
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", 
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    
    return model

def get_transform():
    """Get the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def remove_background(image: Image.Image, model, transform, device) -> Image.Image:
    """
    Remove background from an image using BiRefNet.
    
    Args:
        image: Input PIL Image
        model: BiRefNet model
        transform: Image transformation pipeline
        device: torch device (cuda/cpu)
    
    Returns:
        PIL Image with transparent background
    """
    original_size = image.size
    image_rgb = image.convert("RGB")
    
    # Preprocess
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Post-process
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_size)
    
    # Apply mask as alpha channel
    image_rgb.putalpha(mask)
    
    return image_rgb

def process_directory(input_dir: str, output_dir: str, device: str = "cuda"):
    """
    Process all images in input directory and save to output directory.
    
    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for processed images
        device: Device to run model on ('cuda' or 'cpu')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup model
    model = setup_model(device)
    transform = get_transform()
    
    # Find all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = Image.open(img_path)
            
            # Remove background
            result = remove_background(image, model, transform, device)
            
            # Save as PNG with original filename
            output_filename = img_path.stem + ".png"
            output_file = output_path / output_filename
            result.save(output_file, "PNG")
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"✓ Successful: {successful}")
    if failed > 0:
        print(f"✗ Failed: {failed}")
    print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Bulk remove backgrounds from images using BiRefNet"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to directory for output images (will be created if it doesn't exist)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run model on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.device)

if __name__ == "__main__":
    main()
