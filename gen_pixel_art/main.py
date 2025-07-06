from pathlib import Path
from PIL import Image
import numpy as np
from gen_pixel_art import colors
from gen_pixel_art import mesh

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
            out[j, i] = colors.get_cell_color(cell)

    return Image.fromarray(out, mode="RGB")

def main():
    data_dir = Path.cwd() / "data"
    img_paths = [
        data_dir / "characters" / "warrior.png",
        data_dir / "characters" / "werewolf.png",
        data_dir / "creatures" / "blob.png",
        data_dir / "creatures" / "bat.png",
        data_dir / "objects" / "treasure.png",
        data_dir / "objects" / "gemstone.png",
        data_dir / "tiles" / "grass.png",
        data_dir / "tiles" / "stone.png",
        ]
    for img_path in img_paths:
        output_dir = Path.cwd() / "output" / img_path.stem
        output_dir.mkdir(exist_ok=True, parents=True)

        img = Image.open(img_path).convert("RGBA")
        img_mesh = mesh.compute_mesh(img, output_dir=output_dir)
        # TODO: automatically detect the number of colors needed
        paletted_img = colors.palette_img(img, num_colors = 24)
        pixelated_img = downsample(paletted_img, img_mesh)
        pixelated_img.save(output_dir / "pixelated.png")

if __name__ == "__main__":
    main()
