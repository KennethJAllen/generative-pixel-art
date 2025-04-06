"""Utility functions for generative pixel art."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

def crop_transparent_edges(image_path: Path, output_path: Path=None):
    """Crops the transparent edges of an image."""
    image = Image.open(image_path).convert("RGBA")
    alpha = image.split()[-1]
    # Get bounding box of non-transparent areas
    bbox = alpha.getbbox()
    if not bbox:
        # Image is fully transparent
        return None

    # Crop image to bounding box
    cropped_image = image.crop(bbox)
    if output_path:
        cropped_image.save(output_path)
    return cropped_image

def compute_mse_region(full_image, test_image, region_box):
    """
    Compute MSE between the sub-region of 'full_image' and 'test_image'
    over the same bounding box (region_box = (left, top, right, bottom)).
    
    Both images should be in the same mode (e.g., RGBA).
    """
    # Crop the relevant region
    region_full = full_image.crop(region_box)
    region_test = test_image.crop(region_box)

    arr_full = np.asarray(region_full, dtype=np.float32)
    arr_test = np.asarray(region_test, dtype=np.float32)

    # Ensure shapes match
    if arr_full.shape != arr_test.shape:
        return np.inf

    return np.mean((arr_full - arr_test) ** 2)

def blur(image: Image.Image):
    """Filters the image to denoise."""
    #image.filter(ImageFilter.BLUR)
    smoothed_image = image.filter(ImageFilter.GaussianBlur(radius=1))
    return smoothed_image

def enhance(image: Image.Image):
    return image.filter(ImageFilter.EDGE_ENHANCE)
