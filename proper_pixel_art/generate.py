from pathlib import Path
from PIL import Image
from proper_pixel_art import colors, mesh, utils

def generate_pixel_art(
        img_path: Path,
        output_dir: Path,
        num_colors: int = 16,
        upsample_factor: int = 2,
        pixel_size: int | None = None,
        transparent_background: bool = False,
        save_intermediates: bool = False,
        ) -> None:
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
        Upsample result by pixel_size factor after algorithm is complete if not None.
    - upsample_factor:
        Upsample original image by this factor. It may help detect lines.
    - transparent_background:
        If True, floos fills each corner of the result with transparent alpha.
        Default False.
    - save_intermediate_imgs:
        If True, saves images of edge detection and mesh generation to output dir.
    """
    img = Image.open(img_path).convert("RGBA")
    work_dir = output_dir / img_path.stem
    work_dir.mkdir(exist_ok=True)

    if save_intermediates:
        mesh_dir = work_dir
    else:
        mesh_dir = None

    mesh_coords, scaled_img = mesh.compute_mesh_with_scaling(img, upsample_factor, output_dir=mesh_dir)

    paletted_img = colors.palette_img(scaled_img, num_colors=num_colors)

    result = colors.downsample(paletted_img, mesh_coords, transparent_background=transparent_background)
    if pixel_size is not None:
        result = utils.scale_img(result, pixel_size)

    result.save(work_dir / "result.png")

def main():
    data_dir = Path.cwd() / "assets"

    img_paths_and_colors = [
        (data_dir / "blob" / "blob.png", 16),
        (data_dir / "bat" / "bat.png", 16),
        (data_dir / "demon" / "demon.png", 64),
        (data_dir / "ash" / "ash.png", 16),
        (data_dir / "pumpkin" / "pumpkin.png", 32),
        (data_dir / "mountain" / "mountain.png", 64),
        ]

    for img_path, num_colors in img_paths_and_colors:
        output_dir = Path.cwd() / "output"
        output_dir.mkdir(exist_ok=True, parents=True)
        generate_pixel_art(img_path,
                           output_dir,
                           pixel_size=20,
                           num_colors=num_colors,
                           transparent_background=False,
                           save_intermediates=True,
                           )

if __name__ == "__main__":
    main()
