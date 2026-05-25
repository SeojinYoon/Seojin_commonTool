
import numpy as np

def rotation_matrix_from_axis_angle(axis, theta):
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis

    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])

    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R
