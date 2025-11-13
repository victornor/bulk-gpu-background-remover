# Bulk GPU Background Remover

A simple Python script for bulk background removal from images using the BiRefNet model. Processes entire directories of images and outputs transparent PNGs, utilizing your local GPU for fast processing.

## Examples

Here are some example results from the background removal process:

<table>
  <tr>
    <td><img src="images/img1.png" alt="Example 1" width="200"/></td>
    <td><img src="images/img2.png" alt="Example 2" width="200"/></td>
    <td><img src="images/img3.png" alt="Example 3" width="200"/></td>
  </tr>
  <tr>
    <td><img src="images/img4.png" alt="Example 4" width="200"/></td>
    <td><img src="images/img5.png" alt="Example 5" width="200"/></td>
    <td><img src="images/img6.png" alt="Example 6" width="200"/></td>
  </tr>
</table>


## Features

- Batch process entire directories of images
- Automatic GPU acceleration (NVIDIA CUDA)
- Supports common image formats: JPG, PNG, BMP, WebP, TIFF
- Progress bar for tracking processing
- Outputs transparent PNG files
- Preserves original image dimensions

## Implementation

This tool uses the [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) (Bilateral Reference Network) model from Hugging Face for high-quality background segmentation. The script:

1. Loads images from an input directory
2. Applies the BiRefNet segmentation model to generate alpha masks
3. Composites the mask with the original image for transparency
4. Saves the results as PNG files in the output directory

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:victornor/bulk-gpu-background-remover.git
cd bulk-gpu-background-remover
```

### 2. Create a virtual environment

```bash
python -m venv bgremove
source bgremove/bin/activate  # On Linux/Mac
# or
bgremove\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you need CUDA-specific PyTorch for your NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python bulk_remove_bg.py <input_directory> <output_directory>
```

Example:

```bash
python bulk_remove_bg.py ./input_images ./output_images
```

Force CPU usage (if GPU is unavailable):

```bash
python bulk_remove_bg.py ./input_images ./output_images --device cpu
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- ~500MB disk space for the BiRefNet model (downloaded on first run)

## Dependencies

- PyTorch & TorchVision
- Hugging Face Transformers
- Pillow (PIL)
- einops, kornia, timm (required by BiRefNet)
- tqdm (progress bars)

See `requirements.txt` for specific versions.

## Example

```bash
# Activate virtual environment
source bgremove/bin/activate

# Process all images in a directory
python bulk_remove_bg.py /path/to/photos /path/to/output

# Output:
# Using GPU: NVIDIA GeForce RTX 5080
# Loading BiRefNet model on cuda...
# Found 150 images to process
# Processing images: 100%|████████████| 150/150 [00:45<00:00, 3.33it/s]
# Processing complete!
# ✓ Successful: 150
# Output saved to: /path/to/output
```

## License

MIT License - feel free to use and modify as needed.

## Credits

- BiRefNet model by [ZhengPeng7](https://huggingface.co/ZhengPeng7/BiRefNet)
- Built with Hugging Face Transformers and PyTorch
