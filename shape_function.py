import numpy as np

from components_shape_function.A import A, dA_dx, dA_dy, d2A_dx2, d2A_dy2, d2A_dydx, inverse_dif
from components_shape_function.B import dB_dx, dB_dy, d2B_dx2, d2B_dy2, B, d2B_dydx
from components_shape_function.p import p, dp_dx, dp_dy, d2p_dx2, d2p_dy2, d2p_dydx


# Функции формы и её вторые производные
def F(point, all_points, w):
    F_result = np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(A(point, w, all_points))), B(point, w, all_points))
    return F_result[0]


def d2Fdx2(point, all_points, w, dwdx, d2wdx2):

    A_matrix = A(point, w, all_points)
    p_T = np.transpose(p(point))
    B_matrix = B(point, w, all_points)

    d2Fdx2_result = np.dot(np.dot(np.transpose(d2p_dx2(point)), np.linalg.inv(A_matrix)), B_matrix) + \
                    np.dot(np.dot(p_T, inverse_dif(d2A_dx2(point, d2wdx2, all_points), A_matrix)),
                           B_matrix) + \
                    np.dot(np.dot(p_T, np.linalg.inv(A_matrix)), d2B_dx2(point, d2wdx2, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(dp_dx(point)), inverse_dif(dA_dx(point, dwdx, all_points), A_matrix)),
                               B_matrix) + \
                    2 * np.dot(np.dot(np.transpose(dp_dx(point)), np.linalg.inv(A_matrix)), dB_dx(point, dwdx, all_points)) + \
                    2 * np.dot(np.dot(p_T, inverse_dif(dA_dx(point, dwdx, all_points), A_matrix)),
                               dB_dx(point, dwdx, all_points))

    return d2Fdx2_result


def d2Fdy2(point, all_points, w, dwdy, d2wdy2):

    A_matrix = A(point, w, all_points)
    p_T = np.transpose(p(point))
    B_matrix = B(point, w, all_points)

    d2Fdy2_result = np.dot(np.dot(np.transpose(d2p_dy2(point)), np.linalg.inv(A_matrix)), B_matrix) + \
                    np.dot(np.dot(p_T, inverse_dif(d2A_dy2(point, d2wdy2, all_points), A_matrix)), B_matrix) + \
                    np.dot(np.dot(p_T, np.linalg.inv(A_matrix)), d2B_dy2(point, d2wdy2, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(dp_dy(point)), inverse_dif(dA_dy(point, dwdy, all_points), A_matrix)),
                               B_matrix) + \
                    2 * np.dot(np.dot(np.transpose(dp_dy(point)), np.linalg.inv(A_matrix)), dB_dy(point, dwdy, all_points)) + \
                    2 * np.dot(np.dot(p_T, inverse_dif(dA_dy(point, dwdy, all_points), A_matrix)), dB_dy(point, dwdy, all_points))

    return d2Fdy2_result


def d2Fdydx(point, all_points, w, dwdx, dwdy, d2wdxdy):

    A_matrix = A(point, w, all_points)
    p_T = np.transpose(p(point))
    B_matrix = B(point, w, all_points)

    d2Fdydx_result = np.dot(np.dot(np.transpose(d2p_dydx(point)), np.linalg.inv(A_matrix)), B_matrix) + \
                     np.dot(np.dot(np.transpose(dp_dx(point)), inverse_dif(dA_dy(point, dwdy, all_points), A_matrix)),
                            B_matrix) + \
                     np.dot(np.dot(np.transpose(dp_dx(point)), np.linalg.inv(A_matrix)), dB_dy(point, dwdy, all_points)) + \
                     np.dot(np.dot(np.transpose(dp_dy(point)), inverse_dif(dA_dx(point, dwdx, all_points), A_matrix)),
                            B_matrix) + \
                     np.dot(np.dot(p_T, inverse_dif(d2A_dydx(point, d2wdxdy, all_points), A_matrix)), B_matrix) + \
                     np.dot(np.dot(p_T, inverse_dif(dA_dx(point, dwdx, all_points), A_matrix)), dB_dy(point, dwdy, all_points)) + \
                     np.dot(np.dot(np.transpose(dp_dy(point)), np.linalg.inv(A_matrix)), dB_dx(point, dwdx, all_points)) + \
                     np.dot(np.dot(p_T, inverse_dif(dA_dy(point, dwdy, all_points), A_matrix)), dB_dx(point, dwdx, all_points)) + \
                     np.dot(np.dot(p_T, np.linalg.inv(A_matrix)), d2B_dydx(point, d2wdxdy, all_points))

    return d2Fdydx_result
