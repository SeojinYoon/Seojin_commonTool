
# Common Libraries
import numpy as np
from scipy.spatial import KDTree

# Functions
def gaussian_weighted_smoothing(coords, values, sigma=1.0):
    """
    Apply Gaussian smoothing to scattered data without using a grid.
    
    Args:
    - coords: (N, 2) array of x, y coordinates.
    - values: (N,) array of corresponding values.
    - sigma: Standard deviation for Gaussian weighting.
    
    Returns:
    - smoothed_values: Smoothed values at each original coordinate.
    """
    tree = KDTree(coords)
    smoothed_values = np.zeros_like(values)
    for i, point in enumerate(coords):
        distances, indices = tree.query(point, k=50)  # Consider 50 nearest neighbors
        weights = np.exp(-distances**2 / (2 * sigma**2))
        smoothed_values[i] = np.sum(values[indices] * weights) / np.sum(weights)
    return smoothed_values

def get_bounding_box(hemisphere, virtual_strip_mask):
    """
    Get bounding box from virtual strip mask

    :param hemisphere(string): "L" or "R"
    :param virtual_strip_mask(np.array): strip mask

    return rect
    """
    template_path = surf_paths(hemisphere)[f"{hemisphere}_template_surface_path"]
    temploate_surface_data = nb.load(template_path)
    vertex_locs = temploate_surface_data.darrays[0].data[:, :2]
    
    rect_vertexes = vertex_locs[np.where(virtual_strip_mask == 1, True, False)]
    min_rect_x, max_rect_x = np.min(rect_vertexes[:, 0]), np.max(rect_vertexes[:, 0])
    min_rect_y, max_rect_y = np.min(rect_vertexes[:, 1]), np.max(rect_vertexes[:, 1])

    left_bottom = (min_rect_x, min_rect_y)
    width = max_rect_x - min_rect_x
    height = max_rect_y - min_rect_y

    return {
        "left_bottom" : left_bottom,
        "width" : width,
        "height" : height,
    }
