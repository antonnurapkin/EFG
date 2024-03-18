import numpy as np


# def p(point):
#     return np.array([[1], [point.x], [point.y]])
#
#
# def dp_dx(point):
#     return np.array([[0], [1], [0]])
#
#
# def dp_dy(point):
#     return np.array([[0], [0], [1]])


def d2p_dx2(point):
    return np.array([[0], [0], [0]])


def d2p_dy2(point):
    return np.array([[0], [0], [0]])


def d2p_dydx(point):
    return np.array([[0], [0], [0]])

def p(point):
    return np.array([[1], [point.x], [point.y], [point.y * point.x], [point.x ** 2], [point.y ** 2]])


def dp_dx(point):
    return np.array([[0], [1], [0], [point.y], [2 * point.x], [0]])


def dp_dy(point):
    return np.array([[0], [0], [1], [point.x], [0], [2 * point.y]])
