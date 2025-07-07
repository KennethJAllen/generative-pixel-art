from pathlib import Path
from PIL import Image
from gen_pixel_art import colors, mesh, utils

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
        data_dir / "game" / "bowser.jpg",
        data_dir / "large" / "demon.png",
        data_dir / "large" / "angel.png",
        data_dir / "game" / "ash.png",
        ]
    for img_path in img_paths:
        output_dir = Path.cwd() / "output" / img_path.stem
        output_dir.mkdir(exist_ok=True, parents=True)

        img = Image.open(img_path).convert("RGBA")
        img_mesh = mesh.compute_mesh(img, output_dir=output_dir)

        #paletted_img = colors.palette_img(img, num_colors = 16)
        paletted_img = colors.auto_quantize(img)
        pixelated_img = colors.downsample(paletted_img, img_mesh)
        pixelated_img.save(output_dir / "result.png")

        # Upscale true pixelated image for deomonstation purposes
        upscaled_img = utils.upscale(pixelated_img)
        upscaled_img.save(output_dir / "upscaled.png")

if __name__ == "__main__":
    main()
