"""Utility functions for generative pixel art."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageChops

def crop_border(image : Image.Image, border: int=1, output_path=None):
    """Crop the boder of an image by a few pixels."""
    width, height = image.size

    # Define box: (left, upper, right, lower)
    cropped = image.crop((
        border,
        border,
        width - border,
        height - border
    ))

    if output_path:
        cropped.save(output_path)

    return cropped

def crop_transparent_edges(image: Image.Image, output_path: Path=None):
    """
    Crops the transparent edges of an image.
    Trim the image by a few pixels
    """
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

def crop_white_edges(image: Image.Image, tolerance: int = 10) -> Image.Image:
    """
    Return `img` cropped so that no pure‑white (255,255,255) border pixels remain.

    Parameters
    ----------
    img : PIL.Image
        The source image.  Mode is converted to RGB if needed.
    tolerance : int, optional
        Accept any channel value ≥ 255 − tolerance as white.
        0 keeps only fully‑white pixels; 10 allows light grays, etc.

    Returns
    -------
    PIL.Image
        Cropped image.  If the whole image is white, returns the original.
    """
    # Ensure three channels for a clean comparison.
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Build a white background the same size.
    bg = Image.new("RGB", image.size, (255, 255, 255))

    # Difference highlights every non‑white pixel.
    diff = ImageChops.difference(image, bg)

    if tolerance:
        # Shift the difference image down so that “almost white”
        # pixels also become zero.
        diff = ImageChops.add(diff, diff, scale=1.0, offset=-tolerance)

    bbox = diff.getbbox()
    cropped_image = image.crop(bbox)
    return cropped_image

def compute_mse(image1: Image.Image, image2: Image.Image, region_box: tuple[int] = None):
    """
    Compute MSE between the sub-region of 'full_image' and 'test_image'
    over the same bounding box (region_box = (left, top, right, bottom)).
    
    Both images should be in the same mode (e.g., RGBA).
    """
    # Crop the relevant region
    if region_box is not None:
        region_full = image1.crop(region_box)
        region_test = image2.crop(region_box)
    else:
        region_full = image1
        region_test = image2

    arr_full = np.asarray(region_full, dtype=np.float32)
    arr_test = np.asarray(region_test, dtype=np.float32)

    # Ensure shapes match
    if arr_full.shape != arr_test.shape:
        raise ValueError("Shapes do not match.")

    return np.mean((arr_full - arr_test) ** 2)

def blur(image: Image.Image) -> Image.Image:
    """Filters the image to denoise."""
    #image.filter(ImageFilter.BLUR)
    smoothed_image = image.filter(ImageFilter.GaussianBlur(radius=1))
    return smoothed_image

def enhance(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.EDGE_ENHANCE)
