"""Reads and processes pixel art."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageChops
import matplotlib.pyplot as plt
from gen_pixel_art import utils

INPUT_DIR = Path(__file__).resolve().parent.parent / 'data'
MAX_PIXEL = 255

def read_images() -> list[Image.Image]:
    """Reads all images in input directory."""
    images = []
    image_files = list(INPUT_DIR.glob('*.png'))
    for image_file in image_files:
        image = Image.open(image_file)
        images.append(image)
    return images

def get_pixel_conversion(image: Image.Image) -> int:
    """Outputs the number of true pixels per image pixel in the high resolution pixel art image."""
    edges = image.filter(ImageFilter.FIND_EDGES).convert("L")
    image_array = np.array(edges)/MAX_PIXEL
    image_array = image_array.round()
    M, N = image_array.shape
    F = np.fft.fft2(image_array)  # Compute 2D Fourier Transform
    F_shifted = np.fft.fftshift(F)  # Shift zero frequency to the center
    magnitude_spectrum = np.abs(F_shifted)

    horizontal_sum = np.sum(magnitude_spectrum, axis=0)
    vertical_sum = np.sum(magnitude_spectrum, axis=1)

    ignore_fraction = 0.8  # You can adjust this to ignore more of the center
    horizontal_sum[:int(N * ignore_fraction)] = 0
    horizontal_sum[-int(N * ignore_fraction):] = 0
    vertical_sum[:int(M * ignore_fraction)] = 0
    vertical_sum[-int(M * ignore_fraction):] = 0

    # Find dominant frequencies (peaks)
    dominant_horizontal_freq = np.argmax(horizontal_sum) - N//2
    dominant_vertical_freq = np.argmax(vertical_sum) - M//2

    if dominant_horizontal_freq != 0:
        wavelength_x = N / abs(dominant_horizontal_freq)
    else:
        wavelength_x = float('inf')  # No horizontal frequency

    if dominant_vertical_freq != 0:
        wavelength_y = M / abs(dominant_vertical_freq)
    else:
        wavelength_y = float('inf')  # No vertical frequency

    # Print the results
    print(f"Estimated pixel width (horizontal): {wavelength_x:.2f} pixels")
    print(f"Estimated pixel height (vertical): {wavelength_y:.2f} pixels")

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Magnitude spectrum (log scale)
    plt.subplot(1, 2, 1)
    plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
    plt.title("Magnitude Spectrum (Log Scale)")
    plt.axis('off')

    # Horizontal and vertical frequency distributions
    plt.subplot(2, 2, 3)
    plt.plot(horizontal_sum)
    plt.title("Horizontal Frequency Distribution")

    plt.subplot(2, 2, 4)
    plt.plot(vertical_sum)
    plt.title("Vertical Frequency Distribution")

    plt.tight_layout()
    plt.show()

def main() -> None:
    """Main access point for the script."""
    images = read_images()
    for image in images:
        smoothed_image = utils.blur(image)
        enhanced_image = utils.enhance(image)
        get_pixel_conversion(image)
        pass
        #diff_image = ImageChops.difference(image, smoothed_image)
        #concatenated_image = concatenate_images([image, smoothed_image, enhanced_image])
        #concatenated_image.show()

        #downscaled_image = downscale(image)
        #downscaled_smooth_image = downscale(smoothed_image)
        #concatenated_downscaled_image = concatenate_images([downscaled_image, downscaled_smooth_image])
        #concatenated_downscaled_image.show()

if __name__ == "__main__":
    main()
