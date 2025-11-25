
# Common Libraries
import sys
import numpy as np

# Custom Libraries
sys.path.append("/home/seojin/Seojin_commonTool/Module")
from sj_math import projection

# Functions
def surface_profile_onUV(data_paths,
                         vertices, 
                         uv_coordinates, 
                         from_point, 
                         to_point, 
                         n_sampling, 
                         width):
    """
    Surface profile on UV coordinates.

    This function calculates surface profile data over UV coordinates by sampling 
    along a defined path and width. It projects UV coordinates onto a specified line, 
    samples data from given image files, and evaluates whether vertices fall within 
    defined coverage areas.

    :param data_paths(list of str): List of file paths to the image data (e.g., NIfTI files).
                       Each file represents an image from which sampling will be performed.
    :param vertices(np.ndarray of shape (n_vertices, 3)): Array of vertices in 3D space representing the surface geometry.
    :param uv_coordinates(np.ndarray of shape (n_vertices, 2)): UV coordinate array of shape (n_vertices, 2).
                           Each row represents the (u, v) coordinate of a vertex.
                           
    :param from_point(np.ndarray of shape (2,)): Starting point of the projection line in UV space.
                       
    :param to_point(np.ndarray of shape (2,)) Ending point of the projection line in UV space.
                     
    :param n_sampling(int): Number of sampling intervals along the projection line.
                       This determines the granularity of sampling.
                       
    :param width(float): Width of the sampling region around the projection line.
                  Vertices within this distance from the line are considered covered.

    :return: A dictionary with the following keys:
             - "sampling_datas": A 2D array where each row corresponds to a data file in 
                                `data_paths` and each column corresponds to a sampling interval.
                                (np.ndarray of shape (len(data_paths), n_sampling))
             - "virtual_strip_mask": A boolean mask array indicating whether each vertex 
                                     falls within the coverage area of any sampling interval.
                                     (np.ndarray of shape (n_vertices,))
             - "sampling_coverages": A boolean 2D array indicating for each sampling interval 
                                     and vertex whether the vertex is covered by that interval.
                                     (np.ndarray of shape (n_sampling, n_vertices))
    """
    total_distance = np.sqrt(np.sum(to_point - from_point) ** 2)

    n_vertex = vertices.shape[0]
    
    # Coverage units
    coverage_interval = 1 / n_sampling
    sampling_units = np.arange(0, 1 + coverage_interval, coverage_interval)
    coverage_units = np.array([(sampling_units[i], sampling_units[i+1]) for i in range(len(sampling_units)) if i < len(sampling_units) - 1])

    # Projection
    on_vector = to_point - from_point
    projection_info = projection(uv_coordinates, 
                                 from_point, 
                                 to_point, 
                                 type_ = "origin_correction")
    scalars = projection_info["scalar"]
    projected_data = projection_info["projected_data"]
    residual_data = projection_info["residual_data"]

    # Sampling coverages
    distance = np.sqrt(np.sum(residual_data ** 2, axis = 1))
    sampling_coverages = np.full((n_sampling, n_vertex), False, dtype=bool)
    for i, (start_unit, end_unit) in enumerate(coverage_units):
        is_within_units = (scalars >= start_unit) & (scalars < end_unit)
        within_distance = distance < width
        is_coverage = is_within_units & within_distance

        sampling_coverages[i, :] = is_coverage

    # Sampling datas
    sampling_datas = np.zeros((len(data_paths), n_sampling))
    for i, path in enumerate(data_paths):
        img = nb.load(path)
        affine = img.affine
        img_array = img.get_fdata()
    
        sampling_data = []
        for is_coverage in sampling_coverages:
            target_image_indices = np.apply_along_axis(lambda coord: reference2imageCoord(coord, affine = affine), 
                                                       axis = 1, 
                                                       arr = vertices[is_coverage])
            value = np.mean(img_array[target_image_indices[:, 0], target_image_indices[:, 1], target_image_indices[:, 2]])
            sampling_data.append(value)
        sampling_datas[i, :] = sampling_data

    # Virtual_stip_mask
    virtual_stip_mask = np.any(sampling_coverages, axis = 0)
    
    result_info = {}
    result_info["sampling_datas"] = sampling_datas
    result_info["virtual_strip_mask"] = virtual_stip_mask
    result_info["sampling_coverages"] = sampling_coverages
    
    return result_info
    