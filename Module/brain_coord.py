
# Common Libraries
import numpy as np

# Functions
def image2referenceCoord(ijk, affine):
    """
    change image coordinate to reference coordinate
    reference coordinate can be scanner coordinate or MNI coordinate...
    
    :param ijk: image coordinate(np.array): image coordinates ex) [0,0,0]
    :param affine: affine matrix(np.array): affine transformation matrix 
    
    :return scanner coordinate(np.array): reference coordinates 
    """
    return np.matmul(affine, np.array([ijk[0], ijk[1], ijk[2], 1]))[0:3]

def img2ref(ijk, affine):
    """
    Change image coordinates to reference coordinates
    (e.g., scanner or MNI coordinates) for multiple input points.

    :param ijk(N x 3 numpy array): image coordinates 
    :param affine(4 x 4 numpy array): affine transformation matrix 
    
    :return (N x 3 numpy array): reference coordinates 
    """
    homogeneous_coords = np.c_[ijk, np.ones(ijk.shape[0])]
    transformed_coords = np.dot(homogeneous_coords, affine.T)
    
    return transformed_coords[:, :3]
    
def reference2imageCoord(xyz, affine):
    """
    change reference coordinate(MNI, RAS+) to image coordinate
    reference coordinate can be scanner coordinate or MNI coordinate...
    
    :param xyz: anatomical coordinate(np.array)
    :param affine: affine matrix(np.array)
    
    return image coordinate(np.array)
    """
    
    result = np.matmul(np.linalg.inv(affine), [xyz[0], xyz[1], xyz[2], 1])[0:3]
    result = np.ceil(result).astype(int) # note: This is ad-hoc process - (np.ceil)
    return result

def LPSp_toRASp(xyz):
    """
    Convert LPS+ coordinate to RAS+(MNI) coordinate
    
    :param xyz: LPS+ coord(list)
    
    return xyz(list)
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    return -x, -y, z

def RASp_toLPSp(xyz):
    """
    Convert RAS+(MNI) coordinate to LPS+ coordinate
    
    :param xyz: RAS+ coord(list)
    
    return xyz(list)
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    return -x, -y, z

def image3d_to_1d(ijk, shape_3d):
    """
    Convert 3d coord to 1d coord
    
    :param ijk: index of image(list or array) ex) [1,2,3]
    :param shape: shape of 3d fmri image(list) ex) [96, 114, 96]
    
    return 1d coord
    """
    i = ijk[0]
    j = ijk[1]
    k = ijk[2]
    
    ni = shape_3d[0]
    nj = shape_3d[1]
    nk = shape_3d[2]
    
    return i * (nj * nk) + j * nk + k

def image1d_to_3d(index, shape_3d):
    """
    Convert 1d coord to 3d coord
    
    :param index: index of 1d image ex) [3]
    :param shape: shape of 3d fmri image(list) ex) [96, 114, 96]
    
    return 3d coord
    """
    ni = shape_3d[0]
    nj = shape_3d[1]
    nk = shape_3d[2]
    
    i = int(index / (nj * nk))
    j = int((index - i * nj * nk) / nk)
    k = (index - i * nj * nk) - (j * nk)
    
    return i, j, k

# Examples
if __name__ == "__main__":
    # image2referenceCoord
    image2referenceCoord([0,0,0], full_mask.affine)
    
    # reference2imageCoord
    reference2imageCoord([0,0,0], full_mask.affine)

