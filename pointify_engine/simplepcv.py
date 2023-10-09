import matplotlib.pyplot as plt
from numba import jit
import matplotlib.colors as mcol
import cv2
from joblib import Parallel, delayed, cpu_count
import numpy as np

def plot_histogram(data, bins=256):
    plt.hist(data.ravel(), bins=bins, color='blue', alpha=0.7)
    plt.title('Histogram of Visibility Values')
    plt.xlabel('Visibility Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


@jit(nopython=True)
def get_tangent_angle(height_diff, distance):
    return height_diff / distance


@jit(nopython=True)
def get_sky_portion_for_direction(image, start_x, start_y, dx, dy, width, height):
    max_tangent = -np.inf
    distance = 0
    x, y = start_x, start_y
    step_size = 1.0  # or another value that makes sense for your data
    while 0 <= x < width and 0 <= y < height:
        if distance > 0:
            tangent = get_tangent_angle(image[int(y), int(x)] - image[int(start_y), int(start_x)], distance)
            max_tangent = max(max_tangent, tangent)
        x += dx * step_size
        y += dy * step_size
        distance += np.sqrt((dx * step_size)**2 + (dy * step_size)**2)
    return np.arctan(max_tangent)

@jit(nopython=True)
def compute_sky_visibility_for_chunk(image, start_row, end_row, num_directions=40):
    height, width = image.shape
    directions = [(np.cos(2 * np.pi * i / num_directions), np.sin(2 * np.pi * i / num_directions)) for i in range(num_directions)]
    sky_visibility = np.zeros((end_row - start_row, width))
    for y in range(start_row, end_row):
        for x in range(width):
            total_angle = 0
            for dx, dy in directions:
                total_angle += get_sky_portion_for_direction(image, x, y, dx, dy, width, height)
            sky_visibility[y - start_row, x] = total_angle / num_directions
    return sky_visibility


def compute_sky_visibility(image, num_directions=40, n_jobs=-1):
    height, width = image.shape

    # If n_jobs is set to -1, use all available cores
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Ensure n_jobs does not exceed image height
    n_jobs = min(n_jobs, height)

    chunk_size = height // n_jobs
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_sky_visibility_for_chunk)(image, i, min(i + chunk_size, height), num_directions) for i in
        range(0, height, chunk_size))

    return np.vstack(results)


def export_results(visibility, vmin, vmax, color_choice, color_factor, standardize=False):
    if standardize:
        vmin = np.percentile(visibility, 1)  # 1st percentile
        vmax = np.percentile(visibility, 99)

    # Normalize data
    normalized_data = (visibility - vmin) / (vmax - vmin)

    def adjust_color(color, color_factor, h_color):
        r, g, b = color

        # Calculate grayscale intensity (average of RGB values)
        intensity = (r + g + b) / 3.0


        # Adjust the coloring factor based on intensity
        adjusted_color_factor = color_factor * (1 - intensity)

        # Apply the adjusted coloring factor to the blue component
        if h_color == 0:
            r += adjusted_color_factor * (1 - r)  # Increase blue but ensure it doesn't exceed 1
            return min(r, 1), g, b  # Ensure doesn't exceed 1
        elif h_color == 1:
            g += adjusted_color_factor * (1 - g)  # Increase blue but ensure it doesn't exceed 1
            return r, min(g,1), b  # Ensure doesn't exceed 1
        elif h_color == 2:
            b += adjusted_color_factor * (1 - b)  # Increase blue but ensure it doesn't exceed 1
            return r, g, min(b,1)  # Ensure doesn't exceed 1

    # Original colors
    colors = [(255, 255, 255), (150, 150, 150), (120, 120, 120), (0, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]

    # Adjust colors
    colors_adjusted = [adjust_color(color, color_factor, color_choice) for color in colors_scaled]

    # Create colormap
    custom_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_adjusted, N=256)

    # Apply a colormap from matplotlib (e.g., 'viridis')
    colored_data = custom_cmap(normalized_data)

    # Convert the RGB data to uint8 [0, 255]
    img = (colored_data[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img_bgr