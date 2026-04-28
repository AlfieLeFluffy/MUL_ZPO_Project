import numpy as np

@staticmethod
def norm1(x):
    return np.linalg.norm(np.concatenate(x).T, 1)

@staticmethod
def norm2(x):
    return np.linalg.norm(np.concatenate(x).T)
