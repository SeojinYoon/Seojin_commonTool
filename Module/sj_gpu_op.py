
import numpy as np
from numba import jit, cuda

@cuda.jit
def outer(v1, v2, out):
    for i, e1 in enumerate(v1):
        for j, e2 in enumerate(v2):
            out[i][j] = e1 * e2

if __name__ == "__main__":
    v1 = np.array([1,2,3])
    v2 = np.array([4,5,6])

    out = cuda.to_device(np.zeros((v1.shape[0], v1.shape[0])))
    outer[1,1](v1, v2, out)

