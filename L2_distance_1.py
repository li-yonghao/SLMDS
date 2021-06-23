# coding=UTF-8
import numpy as np


def L2_distance_1(a, b):
    if np.size((a, 0)) == 1:
        a = [a, np.zeros((1, np.size((a, 1))))]
        b = [b, np.zeros((1, np.size((b, 1))))]
    aa = np.sum(np.multiply(a, a), 0)
    bb = np.sum(np.multiply(b, b), 0)
    ab = np.dot(a.T,b)
    d = np.tile(aa.reshape(aa.shape[0],1), (1,np.size(bb),)) + np.tile(bb, (np.size(aa),1)) - 2 * ab
    d = np.real(d)
    d = np.maximum(d, 0)
    # d = np.multiply(d, 1 - np.eye(np.size(d, 0)))
    return d