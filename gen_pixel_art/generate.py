from pathlib import Path
from PIL import Image
from gen_pixel_art import colors, mesh, utils

def generate_pixel_art(img_path: Path, output_dir: Path, num_colors: int = 16, save_intermediate: bool = True) -> None:
        img = Image.open(img_path).convert("RGBA")
        if save_intermediate:
            img_mesh = mesh.compute_mesh(img, output_dir=output_dir)
        else:
            img_mesh = mesh.compute_mesh(img, output_dir=None)

        paletted_img = colors.palette_img(img, num_colors=num_colors)
        pixelated_img = colors.downsample(paletted_img, img_mesh)
        pixelated_img.save(output_dir / "result.png")

        if save_intermediate:
            # Upscale true pixelated image for deomonstation purposes
            upscaled_img = utils.upscale(pixelated_img)
            upscaled_img.save(output_dir / "upscaled.png")

def main():
    data_dir = Path.cwd() / "data"

    num_colors = 16

    img_paths = [
        data_dir / "characters" / "warrior.png",
        data_dir / "characters" / "werewolf.png",
        data_dir / "creatures" / "blob.png",
        data_dir / "creatures" / "bat.png",
        data_dir / "objects" / "treasure.png",
        data_dir / "objects" / "gemstone.png",
        data_dir / "tiles" / "grass.png",
        data_dir / "tiles" / "stone.png",
        data_dir / "game" / "bowser.jpg",
        data_dir / "large" / "demon.png",
        data_dir / "game" / "ash.png",
        data_dir / "game" / "pumpkin.png",
        ]

    for img_path in img_paths:
        output_dir = Path.cwd() / "output" / img_path.stem
        output_dir.mkdir(exist_ok=True, parents=True)
        generate_pixel_art(img_path,
                           output_dir,
                           num_colors=num_colors,
                           save_intermediate=True)

if __name__ == "__main__":
    main()
