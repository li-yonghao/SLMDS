# coding=UTF-8
import numpy as np

def EProjSimplex_new(v):
    k = 1
    global v1

    ft = 1
    n = len(v)
    #v = np.matrix(v)
    v0 = v - np.mean(v) + k / float(n)

    vmin = np.min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 10e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - float(f) / g
            ft = ft + 1
            if ft > 100:
                x = np.maximum(v1, 0)
                break
        x = np.maximum(v1, 0)
    else:
        x = v0
    return x