import numpy as np


def p(point):
    return np.array([[1], [point.x], [point.y]])


def dp_dx(point):
    return np.array([[0], [1], [0]])


def dp_dy(point):
    return np.array([[0], [0], [1]])


# def p(point):
#     return np.array([[1], [point.x], [point.y], [point.y * point.x], [point.x ** 2], [point.y ** 2]])
#
#
# def dp_dx(point):
#     return np.array([[0], [1], [0], [point.y], [2 * point.x], [0]])
#
#
# def dp_dy(point):
#     return np.array([[0], [0], [1], [point.x], [0], [2 * point.y]])


# def p(point):
#     return np.array([[1], [point.x], [point.y], [point.x ** 2], [point.y * point.x], [point.y ** 2], [point.x ** 3], [(point.x ** 2) * point.y],  [(point.y ** 2) * point.x], [point.y ** 3]])
#
#
# def dp_dx(point):
#     return np.array([[0], [1], [0], [2 * point.x], [point.y], [0], [3 * (point.x ** 2)], [2 * point.x * point.y], [point.y ** 2], [0]])
#
#
# def dp_dy(point):
#     return np.array([[0], [0], [1], [0], [point.x], [2 * point.y], [0], [point.x ** 2], [2 * point.y * point.x], [3 * (point.y ** 2)]])
