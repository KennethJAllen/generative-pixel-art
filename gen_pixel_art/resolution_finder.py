from pathlib import Path
from PIL import Image
from gen_pixel_art import utils

def estimate_scale_factor_with_offset(image, max_scale=100):
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
    for scale in range(2, max_scale + 1):
        if scale > width or scale > height:
            break

        # For each offset in [0, k)
        for offset_y in range(scale):
            for offset_x in range(scale):

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
                mse_val = utils.compute_mse_region(image, test_image, region_box)

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
    img_with_alpha = Image.open(image_path).convert("RGBA")
    best_scale, (best_ox, best_oy), best_err, best_img = estimate_scale_factor_with_offset(img_with_alpha)
    print(f"Best Scale Factor: {best_scale}")
    print(f"Best Offsets: offset_x={best_ox}, offset_y={best_oy}")
    print(f"Minimum MSE: {best_err:.2f}")
    output_dir = Path.cwd() / 'output' / 'downscale'
    output_dir.mkdir(exist_ok=True, parents=True)
    downscaled_img_name = "downscaled_" + asset
    best_img.save(output_dir / downscaled_img_name)

if __name__ == "__main__":
    main()
