import numpy as np
import math
from numba import jit



def lumpiness(x, width):
    nr = len(x)
    if nr < 2 * width:
        return 0.0
    lo = range(0, nr, width)
    up = range(width, nr + width, width)
    n_segments = math.ceil(nr / width)
    varx = [np.var(x[lo[idx]:up[idx]]) for idx in range(0, n_segments)]
    return np.var(varx)

def stability(x, width):
    return 0