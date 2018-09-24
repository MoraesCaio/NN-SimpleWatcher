import numpy as np


class Filter:

    kernels = {
        'gx':        np.array([[1,  0, -1], [2, 0, -2], [1,  0, -1]], dtype='int64'),
        'gy':        np.array([[1,  2,  1], [0, 0,  0], [-1, -2, -1]], dtype='int64'),
        'sharpen3':  np.array([[0, -1,  0], [-1, 5, -1], [0, -1,  0]], dtype='int64'),
    }

    kernels_float = {
        'mean3':  np.ones([3] * 2, dtype='float64') / 3 ** 2,
        'mean5':  np.ones([5] * 2, dtype='float64') / 5 ** 2,
        'gaussian3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float64') / 16,
        'gaussian5': np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24,   36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype='float64') / 256,
        'unsharp5':  np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype='float64') / -256,
    }
