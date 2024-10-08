import numpy as np


'''ПАРАМЕТРЫ МАТЕРИАЛА'''
mu = 0.3
E = 2e11
G = E / (2 * (1 + mu))

D_init_array = [[1, mu, 0],
                [mu, 1, 0],
                [0, 0, (1 - mu) / 2]]

D_init_const = E / (1 - mu ** 2)

D = D_init_const * np.array(D_init_array)


'''ПАРАМЕТРЫ ГЕОМЕТРИИ'''
A = 1  # Размер пластины в направлении оси Х
B = 1  # Размер пластины в направлении оси Y
CRACK_LENGTH = B / 10
CRACK_HALF_WIDTH = B / 200


'''ПАРАМЕТРЫ АППРОКСИМАЦИИ'''
METHOD = "Lagrange"  # Penalty

NODES_NUMBER_ON_BOUND_X = 21
NODES_NUMBER_ON_BOUND_Y = 21
NODES_ON_CRACK = 30
CELLS_NUMBER = 24

PENALTY = E * 4e5

ALPHA_X = 2  # Размер области поддержки
DC_X = A / (NODES_NUMBER_ON_BOUND_X - 1)  # характеристическая длина ( расстояние между двумя узлами )

DS = ALPHA_X * DC_X

WEIGHT_FUNCTION_TYPE = "cubic"  # quadratic

LAMBDA = 2


'''НАГРУЗКА'''
P = 10000


'''Трещина'''
x1 = 0
y1 = B / 2

x2 = CRACK_LENGTH
y2 = B / 2

K = ((y2 - y1) / (x2 - x1))
CONST = y1 - ((y2 - y1) / (x2 - x1)) * x1

