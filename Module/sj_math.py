
from scipy.spatial.distance import euclidean
import numpy as np

def round_down(value, decimals):
    factor = 1 / (10 ** decimals)
    return (value // factor) * factor

def digit_length(n):
    return int(math.log10(n)) + 1 if n else 0

def discrete_frechet(P, Q):
    """
    Computes the Discrete Fréchet Distance between two curves P and Q.
    P and Q are arrays of points in the format [x, y].
    
    :param P(list): curve P
    :param Q(list): curve Q
    
    return (float)
    """
    n = len(P)
    m = len(Q)
    ca = np.full((n, m), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = euclidean(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), euclidean(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), euclidean(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), euclidean(P[i], Q[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return c(n - 1, m - 1)

def projection(data, on_vector):
    """
    Projection data on on_vector
    
    :param data(np.array - shape: (#data, #component))
    :param on(np.array - shape: (#component))

    return: 
        tuple
            - scalar(np.array - shape(: #data))
            - projected vector consisting of components
    """
    norm = np.linalg.norm(on_vector)

    dot = np.dot(data, on_vector)
    scalar = dot / (norm ** 2)
    
    return scalar, np.outer(scalar, on_vector)
    
if __name__ == "__main__":
    round_down(0.011, 2)

    digit_length(30)

    a = np.array([
        [1,2,3],
        [1,2,4],
    ])

    projection(a, [1,2,3])
    