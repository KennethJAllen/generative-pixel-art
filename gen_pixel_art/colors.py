from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

def rgba_to_masked_rgb(
        image: Image.Image,
        alpha_threshold: int = 128
        ) -> Image.Image:
    """
    Convert an RGBA image to RGB, 
    setting any pixel with alpha < alpha_threshold to black.
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[..., :3].copy()
    alpha = arr[..., 3]

    # 3. Zero out any pixel whose alpha is below the threshold
    mask = alpha < alpha_threshold
    rgb[mask] = 0 # sets R, G, B = 0 for masked pixels
    return Image.fromarray(rgb, mode="RGB")

def get_cell_color(cell_pixels: np.ndarray) -> tuple[int,int,int]:
    """
    cell_pixels: shape (H_cell, W_cell, 3), dtype=uint8
    returns the most frequent RGB tuple in that block
    """
    # flatten to tuple of pixel values
    flat = list(map(tuple, cell_pixels.reshape(-1, 3)))
    return Counter(flat).most_common(1)[0][0]

def palette_img(img: Image.Image, num_colors: int = 24, quantize_method: int = 1) -> Image.Image:
    rbg_img = rgba_to_masked_rgb(img)
    paletted = rbg_img.quantize(colors=num_colors, method=quantize_method)
    return paletted

def downsample(image: Image.Image, mesh: tuple[list[int], list[int]]) -> Image.Image:
    lines_x, lines_y = mesh
    rgba = image.convert("RGBA")
    # reuse your global-quantized RGB version here:
    rgb = np.array(rgba)[:, :, :3]
    h_new, w_new = len(lines_y) - 1, len(lines_x) - 1
    out = np.zeros((h_new, w_new, 3), dtype=np.uint8)

    for j in range(h_new):
        for i in range(w_new):
            x0, x1 = lines_x[i], lines_x[i+1]
            y0, y1 = lines_y[j], lines_y[j+1]
            cell = rgb[y0:y1, x0:x1]
            out[j, i] = get_cell_color(cell)

    return Image.fromarray(out, mode="RGB")

def main():
    img_path = Path.cwd() / "data" / "objects" / "treasure.png"
    img = Image.open(img_path).convert("RGBA")
    paletted = palette_img(img)
    paletted.show()

if __name__ == "__main__":
    main()
