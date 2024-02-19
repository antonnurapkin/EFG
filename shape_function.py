import numpy as np

from components_shape_function.A import A, dA_dx, dA_dy, inverse_dif
from components_shape_function.B import dB_dx, dB_dy, B
from components_shape_function.p import p, dp_dx, dp_dy


# Функция форма и её производные
def F(point, all_points, w):
    F_result = np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(A(point, w, all_points))), B(point, w, all_points))
    return F_result[0]


def dFdx(point, all_points, w, dwdx):

    A_matrix = A(point, w, all_points)
    p_T = np.transpose(p(point))
    B_matrix = B(point, w, all_points)

    dFdx_result = np.dot(np.dot(np.transpose(dp_dx(point)), np.linalg.inv(A_matrix)), B_matrix) + \
                    np.dot(np.dot(p_T, inverse_dif(dA_dx(point, dwdx, all_points), A_matrix)), B_matrix) +\
                    np.dot(np.dot(p_T, np.linalg.inv(A_matrix)), dB_dx(point, dwdx, all_points))

    return dFdx_result


def dFdy(point, all_points, w, dwdy):

    A_matrix = A(point, w, all_points)
    p_T = np.transpose(p(point))
    B_matrix = B(point, w, all_points)

    dFdy_result = np.dot(np.dot(np.transpose(dp_dy(point)), np.linalg.inv(A_matrix)), B_matrix) + \
                  np.dot(np.dot(p_T, inverse_dif(dA_dy(point, dwdy, all_points), A_matrix)), B_matrix) + \
                  np.dot(np.dot(p_T, np.linalg.inv(A_matrix)), dB_dy(point, dwdy, all_points))

    return dFdy_result