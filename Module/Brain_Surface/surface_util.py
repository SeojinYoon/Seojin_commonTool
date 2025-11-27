
# Common Libraries
import numpy as np
import nibabel as nb
from scipy.spatial import KDTree

# Custom Libraries
from surface_data import surf_paths, load_surfData_fromVolume, vol_to_surf

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

def mean_datas(volume_data_paths: list, 
               hemisphere: str, 
               is_do_smoothing: bool = False,
               sigma: float = 2.0, 
               depths: list = [0,0.2,0.4,0.6,0.8,1.0]):
    """
    average all datas on surface data & do smoothing

    :param volume_data_paths: volume data path(.nii)
    :param hemisphere(string): "L" or "R"
    :param is_do_smoothing: Flag to do smoothing after mean process
    :param sigma: standard deviation for Gaussian weighting
    :param depths(list): Depths of points along line at which to map (0=white/gray, 1=pial). ex) [0.0,0.2,0.4,0.6,0.8,1.0]
    """
    surf_data = load_surfData_fromVolume(volume_data_paths, hemisphere = hemisphere, depths = depths)

    template_path = surf_paths(hemisphere)[f"{hemisphere}_template_surface_path"]
    temploate_surface_data = nb.load(template_path)
    vertex_locs = temploate_surface_data.darrays[0].data[:, :2]

    mean_data = np.mean(surf_data, axis = 1)
    if is_do_smoothing:
        return_data = gaussian_weighted_smoothing(coords = vertex_locs, 
                                                  values = mean_data, 
                                                  sigma = sigma)
    else:
        return_data = mean_data
    return return_data

if __name__ == "__main__":
    template_path = '/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.flat.surf.gii'
    temploate_surface_data = nb.load(template_path)
    vertex_locs = temploate_surface_data.darrays[0].data[:, :2]

    nii_path = "/mnt/ext1/seojin/temp/stat.nii"
    pial_surf_path = '/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.pial.surf.gii'
    white_surf_path = '/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.white.surf.gii'
    surface_data = vol_to_surf(volume_data_path = nii_path,
                               pial_surf_path = spial_surf_path,
                               white_surf_path = white_surf_path,
                               depths = [0,0.2,0.4,0.6,0.8,1.0])
    smoothed_data = gaussian_weighted_smoothing(coords = vertex_locs, 
                                                values = surface_data, 
                                                sigma = 2)

    l_virtual_stip_mask = np.load("/home/seojin/temp/l_virtual_strip_mask.npy")
    get_bounding_box("L", l_virtual_stip_mask)
    