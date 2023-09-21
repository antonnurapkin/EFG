import numpy as np
from params import m, n, step_x, step_y, n_x, n_y


# TODO: Изменить вторую производную весовой функции
def print_coords(array):
    for i in range(n_x):
        for j in range(n_y):
            print((array[i][j].x, array[i][j].y))


# Весовая функция и её производные
def weight_func(x, x_i, alpha_s = 3):

    d_c = (step_x ** 2 + step_y ** 2 )**0.5

    d_s = alpha_s * d_c
    r = abs(x - x_i) / d_s

    if 0 <= r <= 0.5:
        return 2 / 3 - 4 * r **2 + 4 * r ** 3
    elif 0.5 < r <= 1:
        return 4 / 3 - 4 * r + 4 * r ** 2 - 4 / 3 * r ** 3
    elif r > 1:
        return 0 
    

def d_weight_func(x, x_i, alpha_s = 2):

    d_c = (step_x ** 2 + step_y ** 2 )**0.5

    d_s = alpha_s * d_c
    r = abs(x - x_i) / d_s

    if 0 <= r <= 0.5:
        return (-8 * r + 12 * r ** 2) * np.sign(x - x_i)
    elif 0.5 < r <= 1:
        return (-4 + 8 * r - 8 * r ** 2) * np.sign(x - x_i)
    elif r > 1:
        return 0


def d2_weight_func(x, x_i, alpha_s = 2):
    
    d_c = (step_x ** 2 + step_y ** 2 )**0.5

    d_s = alpha_s * d_c
    r = abs(x - x_i) / d_s

    if 0 <= r <= 0.5:
        return (-8 + 24 * r) * np.sign(x - x_i)
    elif 0.5 < r <= 1:
        return (8 - 8 * r) * np.sign(x - x_i)
    elif r > 1:
        return 0 


# Матрицы необходимые для построения функции формы
def p(point):
    return np.array([[1],[p.x],[p.y]]).T
    

def A(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    any_point = point
    for point in all_points:
        point_i = point
        A_local += weight_func(x=any_point.x, x_i=point_i.x) * weight_func(x=any_point.y, x_i=point_i.y)\
              * np.array([[1,            point_i.x,              point_i.y],
                        [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                        [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
        
    return np.array(A_local)


def B(point, all_points):
    B_local = np.zeros((m + 2, n))
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            B_local[0, k] = 1 * weight_func(point=point, point_i=all_points[i][j])
            B_local[1, k] = all_points[i][j].x * weight_func(point=point, point_i=all_points[i][j])
            B_local[2, k] = all_points[i][j].y * weight_func(point=point, point_i=all_points[i][j])
            k += 1

    return np.array(B_local)


# Первые производные матрицы А
def dA_dx(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i][j]
            if point != point_i:
                A_local += d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)\
                      * np.array([[1,            point_i.x,              point_i.y],
                                [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                                [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


def dA_dy(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i][j]
            if point != point_i:
                A_local += weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)\
                      * np.array([[1,            point_i.x,              point_i.y],
                                [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                                [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


# Вторые производные матрицы А
def d2A_dx2(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i][j]
            if point != point_i:
                A_local += d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)\
                      * np.array([[1,            point_i.x,              point_i.y],
                                [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                                [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


def d2A_dy2(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i][j]
            if point != point_i:
                A_local += weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y)\
                      * np.array([[1,            point_i.x,              point_i.y],
                                [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                                [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


def d2A_dydx(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i][j]
            if point != point_i:
                A_local += d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)\
                      * np.array([[1,            point_i.x,              point_i.y],
                                [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                                [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


# Первые производные матрицы B
def dB_dx(point, all_points):
    B_local = np.zeros((m + 2, n))
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i]
            B_local[0, k] = 1 * d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
            B_local[1, k] = all_points[i][j].x * d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
            B_local[2, k] = all_points[i][j].y * d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
            k += 1

    return np.array(B_local)


def dB_dy(point, all_points):
    B_local = np.zeros((m + 2, n))
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i]
            B_local[0, k] = 1 * weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
            B_local[1, k] = all_points[i][j].x * weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
            B_local[2, k] = all_points[i][j].y * weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
            k += 1

    return np.array(B_local)


# Вторые производные матрицы А
def d2B_dx2(point, all_points):
    B_local = np.zeros((m + 2, n))
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i]
            B_local[0, k] = 1 * d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
            B_local[1, k] = all_points[i][j].x * d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
            B_local[2, k] = all_points[i][j].y * d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
            k += 1

    return np.array(B_local)


def d2B_dy2(point, all_points):
    B_local = np.zeros((m + 2, n))
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i]
            B_local[0, k] = 1 * weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y)
            B_local[1, k] = all_points[i][j].x * weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y)
            B_local[2, k] = all_points[i][j].y * weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y)
            k += 1

    return np.array(B_local)


def d2B_dydx(point, all_points):
    B_local = np.zeros((m + 2, n))
    k = 0
    for i in range(n_x):
        for j in range(n_y):
            point_i = all_points[i]
            B_local[0, k] = 1 * d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
            B_local[1, k] = all_points[i][j].x * d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
            B_local[2, k] = all_points[i][j].y * d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
            k += 1

    return np.array(B_local)


# Первые производные вектора p
def dp_dx(point):
    return np.array([[0],[1],[0]])


def dp_dy(point):
    return np.array([[0],[0],[1]])


# Вторые производные вектора p
def d2p_dx2(point):
    return np.array([[0],[0],[0]])


def d2p_dy2(point):
    return np.array([[0],[0],[0]])


def d2p_dydx(point):
    return np.array([[0],[0],[0]])


# Функции формы и её вторые производные
def F(point):
    F_result = np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(A(point))),B(point))
    return F_result

# TODO: Добавить all_points в аргументы
def d2Fdx2(point, all_points):
    d2Fdx2_result = np.dot(np.dot(np.transpose(d2p_dx2(point)),np.linalg.inv(A(point, all_points))),B(point, all_points)) +\
                    np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(d2A_dx2(point, all_points))),B(point, all_points)) +\
                    np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(A(point, all_points))),d2B_dx2(point, all_points)) +\
                    2 * np.dot(np.dot(np.transpose(dp_dx(point)),np.linalg.inv(dA_dx(point, all_points))),B(point, all_points)) +\
                    2 * np.dot(np.dot(np.transpose(dp_dx(point)),np.linalg.inv(A(point, all_points))),dB_dx(point, all_points)) +\
                    2 * np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(dA_dx(point, all_points))),dB_dx(point, all_points))
    
    return d2Fdx2_result


def d2Fdy2(point, all_points):
    d2Fdy2_result = np.dot(np.dot(np.transpose(d2p_dy2(point)),np.linalg.inv(A(point, all_points))),B(point, all_points)) +\
                    np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(d2A_dy2(point, all_points))),B(point, all_points)) +\
                    np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(A(point, all_points))),d2B_dy2(point, all_points)) +\
                    2 * np.dot(np.dot(np.transpose(dp_dy(point)),np.linalg.inv(dA_dy(point, all_points))),B(point, all_points)) +\
                    2 * np.dot(np.dot(np.transpose(dp_dy(point)),np.linalg.inv(A(point, all_points))),dB_dy(point, all_points)) +\
                    2 * np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(dA_dx(point, all_points))),dB_dy(point, all_points))
    
    return d2Fdy2_result


def d2Fdydx(point, all_points):
    # d2Fdydx_result = np.dot(np.dot(np.transpose(d2p_dy2(point)),np.linalg.inv(A(point, all_points))),B(point, all_points)) +\
    #                 np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(d2A_dy2(point, all_points))),B(point, all_points)) +\
    #                 np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(A(point, all_points))),d2B_dy2(point, all_points)) +\
    #                 2 * np.dot(np.dot(np.transpose(dp_dy(point)),np.linalg.inv(dA_dy(point, all_points))),B(point, all_points)) +\
    #                 2 * np.dot(np.dot(np.transpose(dp_dy(point)),np.linalg.inv(A(point, all_points))),dB_dy(point, all_points)) +\
    #                 2 * np.dot(np.dot(np.transpose(p(point)),np.linalg.inv(dA_dx(point, all_points))),dB_dy(point, all_points))
    d2Fdydx_result = None
    return d2Fdydx_result
