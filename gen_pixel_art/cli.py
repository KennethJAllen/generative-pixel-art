import argparse
from pathlib import Path
from gen_pixel_art import generate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a true-resolution pixel-art image from a source image."
    )
    parser.add_argument(
        "-i", "--input",
        dest="image_path",
        type=Path,
        required=True,
        help="Path to the source image file."
    )
    parser.add_argument(
        "-c", "--colors",
        dest="num_colors",
        type=int,
        required=True,
        help="Number of colors to quantize the image to."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        type=Path,
        required=True,
        help="Path where the pixelated image will be saved."
    )

    parser.add_argument(
            "-s", "--save-steps",
            dest="save_steps",
            action="store_true",
            help="Save intermediate images (quantized and pixelated) in the output directory."
        )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    generate.generate_pixel_art(args.image_path,
                                args.output_path,
                                args.num_colors,
                                save_intermediate=args.save_steps)

if __name__ == "__main__":
    main()
