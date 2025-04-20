"""Utility functions for generative pixel art."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation

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
    Return `img` cropped so that no pure-white (255,255,255) border pixels remain.

    Parameters
    ----------
    img : PIL.Image
        The source image.  Mode is converted to RGB if needed.
    tolerance : int, optional
        Accept any channel value ≥ 255 − tolerance as white.
        0 keeps only fully-white pixels; 10 allows light grays, etc.

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

    # Difference highlights every non-white pixel.
    diff = ImageChops.difference(image, bg)

    if tolerance:
        # Shift the difference image down so that “almost white”
        # pixels also become zero.
        diff = ImageChops.add(diff, diff, scale=1.0, offset=-tolerance)

    bbox = diff.getbbox()
    cropped_image = image.crop(bbox)
    return cropped_image

def complete_alpha(image: Image.Image, threshold: int = 255) -> Image.Image:
    """
    Turns pixels with transparency less than the threhold fully transparent.
    Pixels with transparency higher than threshold are turned opaque.
    """
    # Convert the image to a NumPy array
    data = np.array(image)

    # Extract the alpha channel (last channel)
    alpha = data[..., 3]

    # Create a mask for semi-transparent pixels.
    # Here we define semi-transparent as any pixel that is not fully opaque (alpha != 255)
    # If you want to treat fully transparent pixels (alpha == 0) differently, you can adjust the condition.
    semi_transparent_mask = (alpha > 0) & (alpha < threshold)

    # Set all semi-transparent pixels to fully transparent
    data[..., 3][semi_transparent_mask] = 0
    data[..., 3][~semi_transparent_mask] = 255

    # Create a new image from the modified array and save it
    result = Image.fromarray(data).convert("RGBA")
    return result

def dialate_alpha(image: Image.Image) -> Image.Image:
    """
    Dilates the alpha of an image.
    If a pixel is adjacent to a pixel with alpha, that pixel will be given alpha as well.
    """
    # Convert image to a NumPy array
    data = np.array(image)

    # Extract the alpha channel
    alpha = data[..., 3]

    # Create a binary mask for fully transparent pixels
    # (change the condition if you need to treat partially transparent values differently)
    transparent_mask = (alpha == 0)

    # Apply binary dilation to the mask.
    # The structure (a 3x3 array of ones) ensures 8-connected neighbor checking.
    dilated_mask = binary_dilation(transparent_mask, structure=np.ones((3, 3)))

    # Set the alpha of all dilated pixels to 0 (fully transparent)
    data[..., 3][dilated_mask] = 0

    # Create a new image from the modified array
    result_image = Image.fromarray(data)
    return result_image

def compute_mse(image1: Image.Image, image2: Image.Image, region_box: tuple[int] = None) -> float:
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

def fourier_spectrum(image: Image.Image) -> np.ndarray:
    """
    Compute the log-magnitude spectrum of an image.

    Parameters
    ----------
    path
        Image file to analyse.
    save
        Optional path.  If given, the routine writes an 8-bit PNG of the
        spectrum to this location.

    Returns
    -------
    np.ndarray
        A 2-D float array in the range [0, 1] that you can pass straight to
        `plt.imshow`.
    """
    # 1. convert to luminance so the FFT runs on one channel.
    img = image.convert("L")

    # 2. Forward FFT, then move the zero frequency to the centre.
    fshift = np.fft.fftshift(np.fft.fft2(img))

    # 3. Use log(1 + |F|) to compress the dynamic range for display.
    magnitude = np.log1p(np.abs(fshift))

    # 4. Scale to [0, 1] for consistent plotting or saving.
    magnitude -= magnitude.min()
    magnitude /= magnitude.max()

    mag_image = Image.fromarray((magnitude * 255).astype(np.uint8))
    mag_image.show()

    return magnitude



def fundamental_period_fft(image: Image.Image, thresh_ratio: float = 0.25) -> int:
    """
    Return (k_x, k_y) — the horizontal and vertical repeat factors.
    If no clear peak is found in one direction, that value is None.
    
    Parameters
    ----------
    image         : PIL.Image  (RGBA or RGB or L)
    thresh_ratio  : float
        Peaks whose height is at least `thresh_ratio * max_peak`
        are considered.  Increase if you get spurious peaks,
        decrease if the true peak is weak.
    """
    # 1. luminance
    g = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = g.shape

    # 2‑D FFT → magnitude spectrum
    F = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    mag = np.abs(F)

    # 2. Collapse to 1‑D spectra along each axis
    col_profile = mag.sum(axis=0)          # horizontal freq content (x peaks)
    row_profile = mag.sum(axis=1)          # vertical   freq content (y peaks)

    # 3. zero‑out a small DC neighbourhood so it never wins
    pad = 3
    centre_x = width // 2
    centre_y = height // 2
    col_profile[centre_x - pad : centre_x + pad + 1] = 0
    row_profile[centre_y - pad : centre_y + pad + 1] = 0

    # 4. keep only the positive‑frequency half (right / bottom side)
    col_half = col_profile[centre_x + 1 :]
    row_half = row_profile[centre_y + 1 :]

    # 5. peak picking
    def first_peak(arr, min_height):
        peaks, props = find_peaks(arr, height=min_height)
        return peaks[0] if len(peaks) else None

    cx_peak = first_peak(col_half, col_half.max() * thresh_ratio)
    cy_peak = first_peak(row_half, row_half.max() * thresh_ratio)

    kx = int(round(width / (2 * cx_peak))) if cx_peak is not None and cx_peak > 0 else None
    ky = int(round(height / (2 * cy_peak))) if cy_peak is not None and cy_peak > 0 else None

    # Assume the image is square scaled
    k = round((kx + ky)/2)
    return k