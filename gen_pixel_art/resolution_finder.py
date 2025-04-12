from pathlib import Path
from PIL import Image
from gen_pixel_art import utils

def estimate_scale_factor(image: Image.Image, max_scale: int=100):
    """
    Estimate the best integer scale factor by:
      1. Downscaling the image with a candidate factor
      2. Upscaling it back to the original size (via nearest neighbor)
      3. Measuring reconstruction error (MSE) against the original

    Parameters:
    -----------
    image : PIL.Image (RGBA)
        The noisy, potentially upscaled pixel-art-like image with transparency.
    max_scale : int
        The maximum scale factor to try (the minimum is 1).

    Returns:
    --------
    best_scale : int
        The scale factor that yields the best reconstruction.
    best_mse : float
        The MSE at the best scale.
    """
    width, height = image.size

    best_scale = 1
    best_mse_val = float('inf')
    best_img = image

    # Loop over candidate scale factors
    for scale in range(2, max_scale + 1):
        if scale > width or scale > height:
            # No point in going beyond the image's dimensions
            break

        # Downscaled dimensions
        down_width  = width  // scale
        down_height = height // scale

        # Skip invalid factors where integer division yields 0
        if down_width < 1 or down_height < 1:
            continue

        # Downscale using BOX (area resampling)
        downscaled = image.resize((down_width, down_height), Image.Resampling.BOX)

        # Upscale back to the original size using nearest-neighbor
        upscaled = downscaled.resize((width, height), Image.Resampling.NEAREST)

        # Compute error
        mse_val = utils.compute_mse(image, upscaled)

        # Track best
        if mse_val < best_mse_val:
            best_mse_val = mse_val
            best_scale = scale
            best_img = downscaled

    return best_scale, best_mse_val, best_img

def estimate_scale_factor_with_offset(image: Image.Image, max_scale: int=100, max_offset: int = 3):
    """
    Estimate the best integer scale factor and (offset_x, offset_y) by:
      1. Cropping a sub-region of the image starting at (offset_x, offset_y)
         whose width and height are multiples of the factor k.
      2. Downscaling -> Upscaling that sub-region.
      3. Comparing the upscaled sub-region to the original sub-region (MSE).
    
    Parameters:
    -----------
    image : PIL.Image
        The noisy or drifted pixel-art-like image (can be RGBA).
    max_scale : int
        The maximum integer scale factor to try (minimum is 1).
    
    Returns:
    --------
    best_scale : int
        The scale factor that yields the best reconstruction.
    best_offset : (int, int)
        (offset_x, offset_y) in [0, k) that gave the best reconstruction.
    best_mse : float
        The MSE at the best (scale, offset).
    """
    width, height = image.size
    
    best_scale = 1
    best_offset = (0, 0)
    best_mse_val = float('inf')
    best_img = None

    # For each possible scale factor
    for scale in range(3, max_scale + 1):
        if scale > width or scale > height:
            break

        # For each offset in [0, k)
        for offset_y in range(-max_offset, max_offset+1):
            for offset_x in range(-max_offset, max_offset+1):

                # Compute the largest sub-region (width, height) that is a multiple of k
                # starting at (offset_x, offset_y).
                region_width = (width - offset_x) // scale * scale
                region_height = (height - offset_y) // scale * scale
                if region_width <= 0 or region_height <= 0:
                    continue

                # Define the bounding box of the region
                left = offset_x
                top = offset_y
                right = offset_x + region_width
                bottom = offset_y + region_height
                region_box = (left, top, right, bottom)

                # Crop the sub-region
                cropped = image.crop(region_box)

                # Downscale (area-based with BOX)
                down_w = region_width // scale
                down_h = region_height // scale
                # If down_w or down_h is 0, skip
                if down_w < 1 or down_h < 1:
                    continue
                downscaled = cropped.resize((down_w, down_h), Image.Resampling.BOX)

                # Upscale back to original sub-region size
                upscaled = downscaled.resize((region_width, region_height), Image.Resampling.NEAREST)

                # Create a blank image of the same size as the original
                # We'll paste the upscaled sub-region at the correct offset
                # so we can compare it exactly to the same region in the original.
                test_image = Image.new(mode=image.mode, size=(width, height))
                test_image.paste(upscaled, box=(left, top))

                # Compute MSE only within the region_box
                mse_val = utils.compute_mse(image, test_image, region_box)

                # Track the best match
                if mse_val < best_mse_val:
                    best_mse_val = mse_val
                    best_scale = scale
                    best_offset = offset_x, offset_y
                    best_img = downscaled

    return best_scale, best_offset, best_mse_val, best_img

def main():
    asset = 'blob.png'
    image_path = Path.cwd() / 'data' / 'creatures' / asset
    image_with_alpha = Image.open(image_path).convert("RGBA")

    # Start with a solid‑white background, then paint the image on top.
    white_bg = Image.new("RGBA", image_with_alpha.size, (255, 255, 255, 255))
    image_rbg = Image.alpha_composite(white_bg, image_with_alpha).convert("RGB")

    image_cropped_edges = utils.crop_border(image_rbg, border=2)
    fully_cropped_image = utils.crop_white_edges(image_cropped_edges)
    blurred_image = utils.blur(fully_cropped_image)

    fft = utils.fourier_spectrum(blurred_image)

    #best_scale, best_err, best_img = estimate_scale_factor(fully_cropped_image)
    best_scale, (best_ox, best_oy), best_err, best_img = estimate_scale_factor_with_offset(blurred_image)
    print(f"Best Scale Factor: {best_scale}")
    print(f"Best Offsets: offset_x={best_ox}, offset_y={best_oy}")
    print(f"Minimum MSE: {best_err:.2f}")
    output_dir = Path.cwd() / 'output' / 'downscale'
    output_dir.mkdir(exist_ok=True, parents=True)
    downscaled_img_name = "downscaled_" + asset
    best_img.save(output_dir / downscaled_img_name)

if __name__ == "__main__":
    main()
