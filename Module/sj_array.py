
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
