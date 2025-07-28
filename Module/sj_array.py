
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

if __name__ == "__main__":
    map_indicies([0,1,2,3,4,5], [2,3,4])
    