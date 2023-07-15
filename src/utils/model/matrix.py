import numpy as np


def proj(u, v):
    return u * np.dot(v, u) / np.dot(u, u)


def GS(V):
    V = 1.0 * V
    U = np.copy(V)
    for i in range(1, V.shape[1]):
        for j in range(i):
            U[:, i] -= proj(U[:, j], V[:, i])
    den = (U ** 2).sum(axis=0) ** 0.5
    E = U / den
    return E