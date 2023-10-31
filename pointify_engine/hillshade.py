import numpy as np
import matplotlib.colors as mcol
import cv2


def optimized_compute_hillshade_for_grid(elevation_grid, cellsize=1, z_factor=1, altitude=45, azimuth=315):
    PI = np.pi

    # Convert angles to radians outside loop to avoid repeated computations
    zenith_rad = (90.0 - altitude) * PI / 180.0
    azimuth_math = (360.0 - azimuth + 90.0) % 360.0
    azimuth_rad = azimuth_math * PI / 180.0

    rows, cols = elevation_grid.shape
    hillshade_matrix = np.zeros_like(elevation_grid, dtype=float)

    # Compute the rate of change in x and y direction for the entire grid
    dz_dx = (np.roll(elevation_grid, shift=-1, axis=1) - np.roll(elevation_grid, shift=1, axis=1)) / (2 * cellsize)
    dz_dy = (np.roll(elevation_grid, shift=-1, axis=0) - np.roll(elevation_grid, shift=1, axis=0)) / (2 * cellsize)

    # Compute slope and aspect for the entire grid
    slope_rad = np.arctan(z_factor * np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_rad[aspect_rad < 0] = 2 * PI + aspect_rad[aspect_rad < 0]

    # Flat terrain adjustments
    aspect_rad[(dz_dx == 0) & (dz_dy == 0)] = 0
    aspect_rad[(dz_dx == 0) & (dz_dy > 0)] = PI / 2
    aspect_rad[(dz_dx == 0) & (dz_dy < 0)] = 2 * PI - PI / 2

    # Compute hillshade for the entire grid
    hillshade_matrix = 255.0 * ((np.cos(zenith_rad) * np.cos(slope_rad)) +
                                (np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)))
    hillshade_matrix[hillshade_matrix < 0] = 0

    return hillshade_matrix

def export_results(visibility, vmin, vmax, color_choice, color_factor, equalize = False):
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


    # Normalize data
    normalized_data = (visibility - vmin) / (vmax - vmin)

    # Original colors
    colors = [(0, 0, 0),(255, 255, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]

    # Adjust colors
    colors_adjusted = [adjust_color(color, color_factor, color_choice) for color in colors_scaled]

    # Create colormap
    custom_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_adjusted, N=256)

    # Apply a colormap from matplotlib (e.g., 'viridis')
    colored_data = custom_cmap(normalized_data)

    # Convert the RGB data to uint8 [0, 255]
    img = (colored_data[:, :, :3] * 255).astype(np.uint8)

    if equalize:
        img = apply_clahe_color(img)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img_bgr


def apply_clahe_color(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Split the LAB image into L, A and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Define the CLAHE algorithm
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to L channel
    clahe_img = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge([clahe_img, a_channel, b_channel])

    # Convert back to RGB color space
    final_img = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)

    return final_img