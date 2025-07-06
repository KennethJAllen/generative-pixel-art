"""Utility functions for generative pixel art."""
from PIL import Image
import numpy as np

def crop_border(image : Image.Image, num_pixels: int=1) -> Image.Image:
    """Crop the boder of an image by a few pixels."""
    width, height = image.size
    box = (num_pixels, num_pixels, width - num_pixels, height - num_pixels)
    cropped = image.crop(box)
    return cropped

def rgba_to_masked_grayscale(image: Image.Image, opacity_threshold: int = 128) -> Image.Image:
    """
    Convert an RGBA image to 8-bit grayscale,
    forcing any semi- or fully-transparent pixel to zero.
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba)

    # get channels and compute luma (Rec. 601)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    luma = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)

    # zero out anything below the opacity threshold
    luma[a < opacity_threshold] = 0

    greyscale = Image.fromarray(luma, mode="L")
    return greyscale
