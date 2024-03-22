import numpy as np
from params import R0, G, mu, P


def u_radial(r, tetta):
    return (P / (4 * G)) * ((1 - mu) / (1 + mu) * r + (R0 ** 2) / r + (
                (4 / (1 + mu)) * ((R0 ** 2) / r) + r - (R0 ** 4) / (r ** 3)) * np.cos(2 * tetta))


def stress_yy(r, tetta):
    return P * (1 - ((R0 ** 2) / (r ** 2)) * ((3 / 2) * np.cos(2 * tetta) + np.cos(4 * tetta)) + (3 / 2) * (
                (R0 ** 2) / (r ** 2)) * np.cos(4 * tetta))
