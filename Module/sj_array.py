
import numpy as np
from scipy import ndimage

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