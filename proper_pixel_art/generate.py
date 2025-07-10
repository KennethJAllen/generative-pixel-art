from pathlib import Path
from PIL import Image
from proper_pixel_art import colors, mesh, utils

def generate_pixel_art(img_path: Path,
                       output_dir: Path,
                       num_colors: int = 16,
                       pixel_size: int = 20,
                       transparent_background: bool = False) -> None:
    """
    Computes the true resolution pixel art image.
    inputs:
    - img_path:
        path of image to compute true resolution art
    - output_dir:
        output directory
    - num_colors:
        The number of colors to use when quantizing the image.
        This is an important parameter to tune,
        if it is too high, pixels that should be the same color will be different colors
        if it is too low, pixels that should be different colors will be the same color
    - pixel_size:
        Size of pixels to upscale result to after algorithm is complete
    - transparent_background:
        If True, floos fills each corner of the result with transparent alpha.
    """
    img = Image.open(img_path).convert("RGBA")
    output_dir = output_dir / img_path.stem
    output_dir.mkdir(exist_ok=True)

    # Try to upsample first. This may help to detect lines.
    upsampled_img = utils.scale_img(img, 2)
    img_mesh = mesh.compute_mesh(upsampled_img, output_dir=output_dir)

    if len(img_mesh[0]) == 2 or len(img_mesh[1]) == 2:
        # If no mesh is found, then use the original image instead.
        img_mesh = mesh.compute_mesh(img, output_dir=output_dir)
        paletted_img = colors.palette_img(img, num_colors=num_colors)
    else:
        paletted_img = colors.palette_img(upsampled_img, num_colors=num_colors)

    result = colors.downsample(paletted_img, img_mesh, transparent_background=transparent_background)
    upsampled_result = utils.scale_img(result, pixel_size)

    result.save(output_dir / "result.png")
    upsampled_result.save(output_dir / "upsampled.png")

def main():
    data_dir = Path.cwd() / "data"

    img_paths_and_colors = [
         (data_dir / "characters" / "warrior.png", 46),
         (data_dir / "characters" / "werewolf.png", 24),
         (data_dir / "creatures" / "blob.png", 16),
         (data_dir / "creatures" / "bat.png", 16),
         (data_dir / "objects" / "treasure.png", 16),
         (data_dir / "objects" / "gemstone.png", 24),
         (data_dir / "tiles" / "grass.png", 16),
         (data_dir / "tiles" / "stone.png", 16),
         (data_dir / "large" / "demon.png", 64),
         (data_dir / "game" / "ash.png", 16),
         (data_dir / "game" / "pumpkin.png", 32),
         (data_dir / "real" / "gnocchi.png", 32),
         (data_dir / "real" / "mountain.png", 64),
         ]

    for img_path, num_colors in img_paths_and_colors:
        output_dir = Path.cwd() / "output"
        output_dir.mkdir(exist_ok=True, parents=True)
        generate_pixel_art(img_path,
                           output_dir,
                           num_colors=num_colors)

if __name__ == "__main__":
    main()
