import numpy as np
from params import t, r0, G, mu


def u_radial(r, tetta):
    return (t[1] / (4 * G)) * ((1 - mu) / (1 - mu) * r + (r0 ** 2) / r + ((4 / (1 + mu)) * ((r0 ** 2) / r) + r - (r0 ** 4) / (r ** 3)) * np.cos(2 * tetta))
