"""Command line interface"""
import argparse
from pathlib import Path
from PIL import Image
from proper_pixel_art import generate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a true-resolution pixel-art image from a source image."
    )
    parser.add_argument(
        "-i", "--input",
        dest="img_path",
        type=Path,
        required=True,
        help="Path to the source image file."
    )
    parser.add_argument(
        "-o", "--output",
        dest="out_path",
        type=Path,
        required=True,
        help="Path where the pixelated image will be saved. Can be either a directory or a file path."
    )
    parser.add_argument(
        "-c", "--colors",
        dest="num_colors",
        type=int,
        default=16,
        help="Number of colors to quantize the image to. From 1 to 256"
    )
    parser.add_argument(
        "-p", "--pixel-size",
        dest="pixel_size",
        type=int,
        default=1,
        help="Width of the 'pixels' in the output image (default: 1)."
    )
    parser.add_argument(
        "-t", "--transparent",
        dest="transparent",
        action="store_true",
        default=False,
        help="Produce a transparent background in the output if set."
    )
    return parser.parse_args()

def resolve_output_path(out_path: Path, input_path: Path, suffix: str = "_pixelated") -> Path:
    """
    If outpath is a directory, make it a file path
    with filename e.g. (input stem)_pixelated.png
    """
    if out_path.suffix:
        return out_path
    filename = f"{input_path.stem}{suffix}.png"
    return out_path / filename

def main() -> None:
    args = parse_args()
    img_path = Path(args.img_path)
    out_path = resolve_output_path(Path(args.out_path), img_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    img = Image.open(img_path)
    pixelated = generate.pixelate(
        img,
        num_colors = args.num_colors,
        pixel_size = args.pixel_size,
        transparent_background = args.transparent
        )

    pixelated.save(out_path)

if __name__ == "__main__":
    main()
