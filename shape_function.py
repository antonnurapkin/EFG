import numpy as np

from EFG.components_shape_function.A import A, dA_dx, dA_dy, d2A_dx2, d2A_dy2, d2A_dydx
from EFG.components_shape_function.B import dB_dx, dB_dy, d2B_dx2, d2B_dy2, B, d2B_dydx
from EFG.components_shape_function.p import p, dp_dx, dp_dy, d2p_dx2, d2p_dy2, d2p_dydx
from params import n_x, n_y


# Функции формы и её вторые производные
def F(point):
    F_result = np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(A(point))), B(point))
    return F_result


def d2Fdx2(point, all_points):
    d2Fdx2_result = np.dot(np.dot(np.transpose(d2p_dx2(point)), np.linalg.inv(A(point, all_points))), B(point, all_points)) + \
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(d2A_dx2(point, all_points))), B(point, all_points)) + \
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(A(point, all_points))), d2B_dx2(point, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(dp_dx(point)), np.linalg.inv(dA_dx(point, all_points))), B(point, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(dp_dx(point)), np.linalg.inv(A(point, all_points))), dB_dx(point, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(dA_dx(point, all_points))), dB_dx(point, all_points))
                    
    return d2Fdx2_result


def d2Fdy2(point, all_points):

    d2Fdy2_result = np.dot(np.dot(np.transpose(d2p_dy2(point)), np.linalg.inv(A(point, all_points))), B(point, all_points)) + \
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(d2A_dy2(point, all_points))), B(point, all_points)) + \
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(A(point, all_points))), d2B_dy2(point, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(dp_dy(point)), np.linalg.inv(dA_dy(point, all_points))), B(point, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(dp_dy(point)), np.linalg.inv(A(point, all_points))), dB_dy(point, all_points)) + \
                    2 * np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(dA_dx(point, all_points))), dB_dy(point, all_points))
    
    return d2Fdy2_result


# TODO: Вычислить
def d2Fdydx(point, all_points):
    d2Fdydx_result = np.dot(np.dot(np.transpose(d2p_dydx(point)), np.linalg.inv(A(point, all_points))), B(point, all_points)) +\
                    np.dot(np.dot(np.transpose(dp_dx(point)), np.linalg.inv(dA_dy(point, all_points))), B(point, all_points)) +\
                    np.dot(np.dot(np.transpose(dp_dx(point)), np.linalg.inv(A(point, all_points))), dB_dy(point, all_points)) +\
                    np.dot(np.dot(np.transpose(dp_dy(point)), np.linalg.inv(dA_dx(point, all_points))), B(point, all_points)) +\
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(d2A_dydx(point, all_points))), dB_dy(point, all_points)) +\
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(dA_dx(point, all_points))), dB_dy(point, all_points)) +\
                    np.dot(np.dot(np.transpose(dp_dy(point)), np.linalg.inv(A(point, all_points))), dB_dx(point, all_points)) + \
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(dA_dy(point, all_points))), dB_dx(point, all_points)) + \
                    np.dot(np.dot(np.transpose(p(point)), np.linalg.inv(A(point, all_points))), d2B_dydx(point, all_points))

    return d2Fdydx_result
