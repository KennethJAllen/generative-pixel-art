from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
from gen_pixel_art import utils

def close_edges(edges: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a morphological closing to fill small gaps in edge map.
    """
    # Use a rectangular kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def cluster_lines(lines: list[int], threshold: int = 4) -> list[int]:
    """Remove lines that are too close to each other by clustering near values"""
    if not lines:
        return []
    lines = sorted(lines)
    clusters = [[lines[0]]]
    for p in lines[1:]:
        if abs(p - clusters[-1][-1]) <= threshold:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    # use the median of each cluster
    return [int(np.median(cluster)) for cluster in clusters]

def detect_grid_lines(edges: np.ndarray,
                      hough_rho: float = 1.0,
                      hough_theta_rad: float = np.deg2rad(1),
                      hough_threshold: int = 100,
                      hough_min_line_len: int = 50,
                      hough_max_line_gap: int = 10,
                      angle_threshold_deg = 15
                     ) -> tuple[list[int], list[int]]:
    """
    - Use Hough line transformation to detect the pixel edges.
    - Only keep lines that are close to vertical or horizontal
    - Cluster the lines so they aren't too close 
    Return:
    - two lists: x-coordinates (vertical lines) and y-coordinates (horizontal lines)
    """
    hough_lines = cv2.HoughLinesP(edges,
                                  hough_rho,
                                  hough_theta_rad,
                                  hough_threshold,
                                  minLineLength=hough_min_line_len,
                                  maxLineGap=hough_max_line_gap)

    height, width = edges.shape
    # Include the sides of the image in lines since they aren't detected by the Hough transform
    lines_x, lines_y = [0, width-1], [0, height-1]
    if hough_lines is None:
        return lines_x, lines_y

    # Loop over all detected lines, only keep the ones that are close to verticle or horizontal
    for x1, y1, x2, y2 in hough_lines[:,0]:
        dx, dy = x2 - x1, y2 - y1
        angle = abs(np.arctan2(dy, dx))
        # vertical if angle > 90- threshold, horizontal if angle < threshold
        if angle > np.deg2rad(90-angle_threshold_deg):
            lines_x.append(round((x1 + x2)/2))
        elif angle < np.deg2rad(angle_threshold_deg):
            lines_y.append(round((y1 + y2)/2))

    unclustered_lines_x = cluster_lines(lines_x)
    unclustered_lines_y = cluster_lines(lines_y)
    return unclustered_lines_x, unclustered_lines_y

def get_pixel_width(lines_x: list[int], lines_y: list[int], trim_outlier_fraction: float = 0.2) -> int:
    dx = np.diff(lines_x)
    dy = np.diff(lines_y)
    gaps = np.concatenate((dx, dy))

    # Filter lower and upper percentile
    low = np.percentile(gaps, 100 * trim_outlier_fraction)
    hi = np.percentile(gaps, 100 * (1 - trim_outlier_fraction))
    middle = gaps[(gaps >= low) & (gaps <= hi)]
    if len(middle) == 0:
        # fallback to median of all gaps
        middle = gaps

    return np.median(middle)

def complete_grid(lines: list[int], pixel_width: int) -> list[int]:
    """
    Given sorted line coords and pixel width,
    further partition those line coordinates to approximately even spacing.
    """
    section_widths = np.diff(lines)
    complete_lines = lines[:-1]
    for index, section_width in enumerate(section_widths):
        # Get number of pixels to partition section width into
        num_pixels = int(np.round(section_width / pixel_width))
        section_pixel_width = section_width / num_pixels
        line_start = lines[index]
        section_lines = [line_start + int(n*section_pixel_width) for n in range(num_pixels)]
        # Replace the start index in completed lines with list of new line coordinates
        # Everything will be unpacked after to maintain indexes
        complete_lines[index] = section_lines

    complete_lines = [line for sublist in complete_lines for line in sublist]
    # Add last line back in because it was excluded earlier
    complete_lines.append(lines[-1])

    return complete_lines

def overlay_grid_lines(
        image: Image.Image,
        lines_x: list[int],
        lines_y: list[int],
        line_color: tuple[int, int, int] = (255, 0, 0),
        line_width: int = 1
        ) -> Image.Image:
    """
    Overlay vertical and horizontal grid lines over image for visualization.

    inputs:
        image: The source PIL image (RGBA or RGB).
        lines_x: Sorted x-coordinates of vertical grid lines.
        lines_y: Sorted y-coordinates of horizontal grid lines.
        line_color: RGB tuple for the line color (default red).
        line_width: Width of the drawn lines in pixels.
    """
    # Ensure we draw on an RGBA canvas
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    w, h = canvas.size
    # Draw each vertical line
    for x in lines_x:
        draw.line([(x, 0), (x, h)], fill=(*line_color, 255), width=line_width)

    # Draw each horizontal line
    for y in lines_y:
        draw.line([(0, y), (w, y)], fill=(*line_color, 255), width=line_width)

    return canvas

def compute_gridlines(img_path: Path,
                      canny_thresholds: tuple[int] = (50, 200),
                      closure_kernel_size: int = 10,
                      save_images: bool = True) -> tuple[list[int]]:
    """
    Finds grid lines of a high resolution noisy image.
    inputs:
        img_path: Path to image
        canny_thresholds: thresholds 1 and 2 for canny edge detection algorithm
        morphological_closure_kernel_size: Kernel size for the morphological closure

    """
    img = Image.open(img_path).convert("RGBA")
    cropped_img = utils.crop_border(img, num_pixels=1)
    grey_img = utils.rgba_to_masked_grayscale(cropped_img)

    # Find edges using Canny edge detection
    edges = cv2.Canny(np.array(grey_img), *canny_thresholds)

    # Close small gaps in edges with morphological closing
    closed_edges = close_edges(edges, kernel_size=closure_kernel_size)

    # Use Hough transform to detect the pixel lines
    lines_x, lines_y = detect_grid_lines(closed_edges)

    # Get the true width of the pixels
    pixel_width = get_pixel_width(lines_x, lines_y)

    # Fill in the gaps between the lines to complete the grid
    lines_x_complete = complete_grid(lines_x, pixel_width)
    lines_y_complete = complete_grid(lines_y, pixel_width)

    if save_images:
        output_dir = Path.cwd() / "output" / img_path.stem
        output_dir.mkdir(exist_ok=True, parents=True)

        edges_img = Image.fromarray(edges, mode="L")
        edges_img.save(output_dir / "edges.png")
        closed_edges_img = Image.fromarray(closed_edges, mode="L")
        closed_edges_img.save(output_dir / "closed_edges.png")

        img_with_lines = overlay_grid_lines(img, lines_x, lines_y)
        img_with_lines.save(output_dir / "lines.png")
        img_with_completed_lines = overlay_grid_lines(img, lines_x_complete, lines_y_complete)
        img_with_completed_lines.save(output_dir / "grid.png")

    return lines_x_complete, lines_y_complete

def main():
    img_path = Path.cwd() / "data" / "objects" / "treasure.png"
    grid_lines = compute_gridlines(img_path)
    print(grid_lines)

if __name__ == "__main__":
    main()
