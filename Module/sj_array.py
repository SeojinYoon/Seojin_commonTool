
import numpy as np
from scipy import ndimage

dataset1 = np.array([
    [
        [1,0,4],
        [1,0,3],
        [0.5, 1, 1],
        [1, 2, 0],
    ],
    [
        [0.5, 0, 1],
        [0.5, 0, 3],
        [1, 0, 4],
        [2, 0, 3],
    ]
])

def dilation_3d(array):
    """
    :param array: 3d array
    
    return array
    """
    # Define a structuring element with diagonal connections
    struct = ndimage.generate_binary_structure(3, 3)

    # Perform morphological dilation on the image using the structuring element
    dilated_image = ndimage.binary_dilation(array, structure=struct)

    return dilated_image


def dilate_2d(array, repetition = 1):
    dilate_struct = ndimage.generate_binary_structure(2, 2)
    for _ in range(repetition):
        array = ndimage.binary_dilation(array, structure = dilate_struct)
    
    return array

def erose_2d(array, repetition = 1):
    erosion_struct = ndimage.generate_binary_structure(2, 2)
    for _ in range(repetition):
        array = ndimage.binary_erosion(array, structure = erosion_struct)
    return array

def find_sign_change_indices(data):
    # Calculate the sign of each element in the data
    signs = np.sign(data)
    
    # Find where the sign changes (where the difference between consecutive signs is not zero)
    sign_changes = np.diff(signs)
    
    # Indices where the sign changes
    sign_change_indices = np.where(sign_changes != 0)[0]
    
    return sign_change_indices

def map_indicies(original_indices, including_indices):
    """
    Maps the indices from `original_indices` to new indices based on whether 
    they are included in `including_indices`. Indices not in `including_indices` 
    are assigned a value of -1.

    :param original_indices(list): A collection of indices (e.g., integers) that represent the original set of indices to be mapped.
    :param including_indices(list): A collection of indices that should be included in the mapping. 

    return (dictioanry): 
            A dictionary where keys are the values from `original_indices`, 
            and values are the corresponding mapped indices. For indices 
            present in `including_indices`, the values will be sequential 
            starting from 0. For indices not in `including_indices`, the value 
            will be -1.
    """
    converted_indexes = []
    converted_index = 0
    for ori_i in original_indices:
        if ori_i in including_indices:
            converted_indexes.append(converted_index)
            converted_index += 1
        else:
            converted_indexes.append(-1)

    result = {}
    for origin_i, converted_i in zip(original_indices, converted_indexes):
        result[origin_i] = converted_i
    
    return result

def get_ACS_explicit_orientation(orientation: str):
    """
    Convert a shorthand human orientation string into explicit axis pairs

    :param orientation: Starting orientation shorthand, e.g., "RAI" or "LPS".
    
    :return: A list of two-character axis pairs, e.g., ['RL', 'AP', 'IS'].
    """
    starts = [ori for ori in orientation]
    
    axis_groups = [get_ACS_axis_group(coord) for coord in starts]
    ends = [group.replace(start, "") for start, group in zip(starts, axis_groups)]
    
    return [f"{start}{end}" for start, end in zip(starts, ends)]
    
def reorient_ACS_array(data: np.ndarray, 
                       current_orient: str, 
                       target_orient: str) -> np.ndarray:
    """
    Re-orient 3D array

    :param data(shape: (#times, #labels, 3)): 3D array
    :param current_orient: current orientation ex) "LPS"
    :param target_orient: orientation to be converted ex) "RAS"

    return reoriented array
    """
    M = get_ACS_orientation_mat(current_orient, target_orient)
    return data @ M.T

def get_ACS_axis_group(char):
    if char in "LR": return "LR"
    if char in "IS": return "IS"
    if char in "PA": return "PA"
    return None

def get_ACS_orientation_mat(current_orient: str, target_orient: str) -> np.ndarray:
    """
    Get transform matrix in ACS (Anatomical Coordinate System)

    :param current_orient: current orientation ex) LPS
    :param target_orient: target orientation to be converted ex) RPS

    return 3x3 transform matrix 
    """
    current_groups = [get_ACS_axis_group(c) for c in current_orient]
    target_groups = [get_ACS_axis_group(c) for c in target_orient]
    
    M = np.zeros((3, 3))
    for target_idx, t_group in enumerate(target_groups):
        current_idx = current_groups.index(t_group)
        
        c_char = current_orient[current_idx]
        t_char = target_orient[target_idx]
        
        M[target_idx, current_idx] = 1 if c_char == t_char else -1
    return M
    
if __name__ == "__main__":
    map_indicies([0,1,2,3,4,5], [2,3,4])

    n_times = 10
    n_labels = 2
    n_coords = 3
    data = np.zeros((n_times, n_labels, n_coords))
    for label_i in range(n_labels):
        for coord_i in range(n_coords):
            data[:, label_i, coord_i] = np.arange(coord_i * 10, coord_i * 10 + 10, 1)
    reorient_array(data, current_ornt = "LPS", target_ornt = "RSA")
    