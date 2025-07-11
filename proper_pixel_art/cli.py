"""Command line interface"""
import argparse
from pathlib import Path
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
        dest="out_dir",
        type=Path,
        required=True,
        help="Path where the pixelated image will be saved."
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
        default=20,
        help="Size of the pixels in the output image (default: 20)."
    )
    parser.add_argument(
        "-t", "--transparent",
        dest="transparent",
        action="store_true",
        default=False,
        help="Produce a transparent background in the output if set."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    img_path = Path(args.img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image path does not exist: {img_path}")
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")
    generate.generate_pixel_art(img_path,
                                out_dir,
                                args.num_colors,
                                pixel_size=args.pixel_size,
                                transparent_background=args.transparent)

if __name__ == "__main__":
    main()
