
# Common Libraries
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Functions
def calc_dtw_cost(trj1: np.ndarray, trj2: np.ndarray) -> float:
    """
    Compute DTW (Dynamic Time Warping) accumulated cost matrix between two multivariate trajectories.

    :param trj1: First trajectory (shape: n: time points, d: dimensions)
    :param trj2: Second trajectory (shape: m: time points, d: dimensions)

    :return: Accumulated cost matrix (shape: n, m)
    """
    # Lengths of the two trajectories
    n, m = len(trj1), len(trj2)

    """
    Step 1: Compute pairwise distance matrix

    dist_matrix[i, j] = distance between trj1[i] and trj2[j]
    For multivariate data, this is Euclidean distance in d-dim space
    """
    dist_matrix = cdist(trj1, trj2, metric='euclidean')

    """
    Step 2: Initialize the Accumulated Cost Matrix with zeros.
    Dimensions are (n+1, m+1) to accommodate boundary conditions.
    """
    cost = np.zeros((n + 1, m + 1))

    # Set first row/column to infinity (except origin)
    # → Forces alignment path to start at (0,0)
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf

    """
    Step 3: Dynamic Programming
    cost[i, j] = local distance + min(previous three moves)
    
    Three possible moves:
        (i-1, j)   → insertion   (stretch trj1)
        (i, j-1)   → deletion    (stretch trj2)
        (i-1, j-1) → match       (align both)
    
    This builds the optimal warping alignment
    """
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = dist_matrix[i - 1, j - 1] + min(
                cost[i - 1, j], # Vertical move (Insertion)
                cost[i, j - 1], # Horizontal move (Deletion)
                cost[i - 1, j - 1] # Diagonal move (Match)
            )

    return cost

def calc_dtw_distance(trj1: np.ndarray, trj2: np.ndarray) -> float:
    """
    Calculate DTW distance between two multivariate trajectories.

    :param trj1: First trajectory (shape: n: time points, d: dimensions)
    :param trj2: Second trajectory (shape: m: time points, d: dimensions)

    :return: distance between trajectories
    """
    n, m = len(trj1), len(trj2)
    cost = calc_dtw_cost(trj1, trj2)
    return cost[n, m]
    
def dtw_2D_traj(trj1: np.array, trj2: np.array) -> list:
    """
    Computes the optimal warping path between two multivariate time series.

    :param trj1: arrays of shape (#time, #coords)
    :param trj2: arrays of shape (#time, #coords)

    :returns: List of tuples representing the aligned indices [(i1, j1), (i2, j2), ...]
    """
    n, m = len(trj1), len(trj2)
    cost = calc_dtw_cost(trj1, trj2)
            
    # Backtrack from the end of the matrix (n, m) to the start (0, 0)
    # to find the path that yields the minimum cumulative cost.
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        
        # Determine which neighbor provided the minimum cost during the forward pass.
        # 0: Diagonal, 1: Vertical, 2: Horizontal
        min_prev = np.argmin([cost[i-1, j-1], cost[i-1, j], cost[i, j-1]])
        
        if min_prev == 0:
            i, j = i-1, j-1
        elif min_prev == 1:
            i -= 1
        else:
            j -= 1
            
    # Return the path in chronological order (from start to end).
    return path[::-1]

def find_medoid_trajectory(trajectories: list) -> np.ndarray:
    """
    Find the medoid trajectory from a list of trajectories using pairwise DTW distance.

    :param trajectories: List of trajectories, where each trajectory is an array of shape (#time, #dim)

    :return medoid_idx: int
        Index of the medoid trajectory (the trajectory with the minimum total DTW distance to all others).
    :return dist_matrix:
        Pairwise DTW distance matrix of shape (n, n), where n is the number of trajectories.
        dist_matrix[i, j] represents the DTW distance between trajectories[i] and trajectories[j].
    """
    n = len(trajectories)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = calc_dtw_distance(trajectories[i], trajectories[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    total_dist = dist_matrix.sum(axis=1)
    medoid_idx = np.argmin(total_dist)

    return medoid_idx, dist_matrix

def warp_trajectory(template_traj: np.ndarray,
                    traj: np.ndarray,
                    path: list) -> np.ndarray:
    """
    Warp the trajectory to match the length of the template trajectory.

    :param template_traj: Reference trajectory defining the target length (shape: [N, D])
    :param traj: Trajectory to be warped (shape: [M, D])
    :param path: Alignment path (e.g., DTW path) as a list of (template_idx, trial_idx) pairs
    
    :return: Warped trajectory aligned to template length (shape: [N, D])
    """
    n_template = len(template_traj)
    
    # Initialize output array with the same length as the template
    warped = np.zeros((n_template, traj.shape[1]))

    # Convert path to numpy array for indexing
    path = np.array(path)
    template_idx = path[:, 0]
    trial_idx = path[:, 1]

    # Iterate over each template index
    for i in range(n_template):
        # Find all trial indices aligned to the current template index
        matched_trial_idx = trial_idx[template_idx == i]

        if len(matched_trial_idx) > 0:
            # Average matched trial points
            warped[i] = traj[matched_trial_idx].mean(axis=0)
        else:
            # Fallback: use the nearest aligned point (rare case)
            nearest = np.argmin(np.abs(template_idx - i))
            warped[i] = traj[trial_idx[nearest]]

    return warped

def warp_features(template_len: int,
                  features: np.ndarray,
                  path: list) -> np.ndarray:
    """
    Warp features to match the template length using the DTW path.

    :param template_len: Length of the template trajectory
    :param features: Trial feature array of shape (T, D)
    :param path: DTW path as [(template_idx, trial_idx), ...]
    
    :return: Warped feature array (shape: (template_len, D))
    """
    warped = np.zeros((template_len, features.shape[1]))

    path = np.array(path)
    template_idx = path[:, 0]
    trial_idx = path[:, 1]

    if trial_idx.max() >= len(features):
        raise ValueError(
            f"DTW path contains trial index {trial_idx.max()}, "
            f"but features has length {len(features)}."
        )

    for i in range(template_len):
        matched_trial_idx = trial_idx[template_idx == i]

        if len(matched_trial_idx) > 0:
            warped[i] = features[matched_trial_idx].mean(axis=0)
        else:
            nearest = np.argmin(np.abs(template_idx - i))
            warped[i] = features[trial_idx[nearest]]

    return warped
    