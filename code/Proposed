# coding=UTF-8
import numpy as np
from skfeature.utility.construct_W import construct_W
from numpy import linalg as LA
from L2_distance_1 import L2_distance_1
from EProjSimplex_new import EProjSimplex_new
eps = 2.2204e-16


def Proposed(X, Y, select_nub, alpha, beta, gamma):
    num, dim = np.shape(X)
    num, label_num = np.shape(Y)

    k = int(label_num * 2 / 3.0)  # TODO:临时使用的值
    W = np.random.rand(dim, label_num)
    V = np.random.rand(label_num, k)
    B = np.random.rand(k, label_num)
    S = np.random.rand(label_num, label_num)

    Ms = np.dot(V, B)
    As = np.diag(np.sum(Ms, 0))
    Ls = As - Ms

    Idd = np.ones((dim, dim))
    iter = 0
    obj = []
    obji = 1

    while 1:
        AA1 = np.dot(X, W) - np.dot(np.dot(Y, V), B)
        XWYVBtemp = np.sqrt(np.sum(np.multiply(AA1, AA1), 1) + eps)
        d1 = 0.5 / XWYVBtemp
        D1 = np.diag(d1.flat)

        AA2 = Y - np.dot(np.dot(Y, V), B)
        YYVBtemp = np.sqrt(np.sum(np.multiply(AA2, AA2), 1) + eps)
        d2 = 0.5 / YYVBtemp
        D2 = np.diag(d2.flat)

        V = np.multiply(V, np.true_divide(
            np.dot(np.dot(np.dot(np.dot(Y.T, D1), X), W), B.T) + np.dot(np.dot(np.dot(Y.T, D2), Y),
                                                                        B.T) + alpha * np.dot(
                np.dot(np.dot(np.dot(np.dot(Y.T, Y), V), B), Ms), B.T) + beta * np.dot(S, B.T),
            np.dot(np.dot(np.dot(np.dot(np.dot(Y.T, D1), Y), V), B), B.T) + np.dot(
                np.dot(np.dot(np.dot(np.dot(Y.T, D2), Y), V), B),
                B.T) + alpha * np.dot(
                np.dot(np.dot(np.dot(np.dot(Y.T, Y), V), B), As), B.T) + beta * np.dot(np.dot(V, B), B.T) + eps))

        B = np.multiply(B, np.true_divide(
            np.dot(np.dot(np.dot(np.dot(V.T, Y.T), D1), X), W) + np.dot(np.dot(np.dot(V.T, Y.T), D2),
                                                                        Y) + alpha * np.dot(
                np.dot(np.dot(np.dot(np.dot(V.T, Y.T), Y), V), B),
                Ms) + beta * np.dot(V.T, S),
            np.dot(np.dot(np.dot(np.dot(np.dot(V.T, Y.T), D1 + D2), Y), V), B) + alpha * np.dot(
                np.dot(np.dot(np.dot(np.dot(V.T, Y.T), Y), V), B), As) + beta * np.dot(np.dot(V.T, V), B) + eps))

        W = np.multiply(W, np.true_divide(np.dot(np.dot(np.dot(np.dot(X.T, D1), Y), V), B) + gamma * W,
                                          np.dot(np.dot(np.dot(X.T, D1), X), W) + gamma * np.dot(Idd, W) + eps))

        VB = np.dot(V, B)
        F = np.dot(np.dot(Y, V), B)
        dist = L2_distance_1(F, F)
        for i in range(label_num):
            vbi = VB[:, i]
            di = dist[i, :]
            ad = vbi - 0.5 * (alpha / float(beta)) * di
            S[i, :] = EProjSimplex_new(ad)

        Ms = (S + S.T) / 2.0
        As = np.diag(np.sum(Ms, 0))
        Ls = As - Ms

        objectives = 2 * np.trace(np.dot(np.dot(AA1.T, D1), AA1)) + 2 * np.trace(
            np.dot(np.dot(AA2.T, D2), AA2)) + alpha * np.trace(
            np.dot(np.dot(np.dot(np.dot(Y, V), B), Ls), np.dot(np.dot(Y, V), B).T)) + beta * pow(LA.norm(S - VB, "fro"),
                                                                                                 2) + gamma * (
                             np.sum(np.abs(np.dot(W, W.T))) + pow(LA.norm(W, "fro"), 2))

        obj.append(objectives)
        cver = abs((objectives - obji) / float(obji))
        obji = objectives
        iter = iter + 1
        if (iter > 2 and (cver < 1e-3 or iter == 1000)):
            break

    return W
