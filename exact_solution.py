import numpy as np
from params import t, R0, G, mu


def u_radial(r, tetta):
    return (t[1, 0] / (4 * G)) * ((1 - mu) / (1 + mu) * r + (R0 ** 2) / r + ((4 / (1 + mu)) * ((R0 ** 2) / r) + r - (R0 ** 4) / (r ** 3)) * np.cos(2 * tetta))