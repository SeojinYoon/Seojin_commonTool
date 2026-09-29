
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

def euler_xyz_to_quat(x_deg=0.0, y_deg=0.0, z_deg=0.0, degrees=True):
    if degrees:
        ax, ay, az = np.radians([x_deg, y_deg, z_deg]) / 2.0
    else:
        ax, ay, az = np.array([x_deg, y_deg, z_deg]) / 2.0

    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return [float(w), float(x), float(y), float(z)]
    